from rev_ai.models import MediaConfig
from rev_ai.streamingclient import RevAiStreamingClient
from google import genai
from google.genai import types
from openai import OpenAI
import pyaudio, json, os, socket, sys, time, ssl
from dotenv import load_dotenv
from mutagen.mp3 import MP3
from datetime import datetime
from time import sleep
from queue import Queue

sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), 'Python-SDK'))
from mistyPy.Robot import Robot
from mistyPy.Events import Events


# ---- macOS + Python 3.10+ SSL workaround for rev_ai's websocket client ----
# The bundled websocket-client library's SNI handshake can fail with
# "SystemError: new style getargs format but argument is not a tuple"
# on some macOS systems. If this patch doesn't resolve the error on your
# machine, switch to the Whisper version of this file.
try:
    import websocket
    def _patched_wrap_sni(sock, sslopt, hostname, check_hostname):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx.wrap_socket(sock, server_hostname=hostname)
    websocket._http._wrap_sni_socket = _patched_wrap_sni
except Exception as e:
    print(f"Warning: could not patch websocket SSL handling: {e}")


# HTTP server port — change this if port 8000 is busy on your machine
HTTP_SERVER_PORT = 8000

# PyAudio recording settings — used for streaming mic input to Rev.ai
AUDIO_RATE = 16000
AUDIO_CHUNK = int(AUDIO_RATE / 10)   # 100ms chunks
SILENCE_TIMEOUT = 2.5                # seconds of silence before we stop listening


# TODO: add 5 more custom actions for the robot
custom_actions = {
    "reset": "IMAGE:e_DefaultContent.jpg; ARMS:40,40,1000; HEAD:-5,0,0,1000;",
    "head-up-down-nod": "IMAGE:e_DefaultContent.jpg; HEAD:-15,0,0,500; PAUSE:500; HEAD:5,0,0,500; PAUSE:500; HEAD:-15,0,0,500; PAUSE:500; HEAD:5,0,0,500; PAUSE:500; HEAD:-5,0,0,500; PAUSE:500;",
    "hi": "IMAGE:e_Admiration.jpg; ARMS:-80,40,100;",
    "listen": "IMAGE:e_Surprise.jpg; HEAD:-6,30,0,1000; PAUSE:2500; HEAD:-5,0,0,500; IMAGE:e_DefaultContent.jpg;"
}


class MistyRobot():

    def __init__(self, misty_ip_address, llm_system_instruction_file):

        # Misty IP address
        self.misty_ip_address = misty_ip_address

        # Misty Robot (Python SDK)
        self.misty = Robot(misty_ip_address)
        self.volume = 30

        # create all of our custom actions
        for action_name, action_script in custom_actions.items():
            self.misty.create_action(
                name = action_name,
                script = action_script,
                overwrite = True
            )

        # Load the Rev.ai access token from the environment variable
        load_dotenv()
        self.revai_access_token = os.getenv('REVAI_ACCESS_TOKEN')
        if not self.revai_access_token:
            raise ValueError("Please set the REVAI_ACCESS_TOKEN environment variable.")

        # variables needed to be initialized for Rev.ai speech-to-text
        self.current_transcript = ""
        self.listening_done = False
        self.user_utterance_counter = 1

        # initialize the OpenAI client for TTS with the OPEN_AI_API_KEY environment variable
        open_ai_api_key = os.getenv('OPEN_AI_API_KEY')
        if not open_ai_api_key:
            raise ValueError("Please set the OPEN_AI_API_KEY environment variable.")
        self.openai_client = OpenAI(api_key=open_ai_api_key)

        # set the path for the robot speech files for both local and robot access
        # this requires that a HTTP server is running in the same directory as this file using
        # the command: python -m http.server <HTTP_SERVER_PORT>
        self.speech_file_path_local = os.path.join(os.path.dirname(__file__), 'robot_speech_files/speech.mp3')
        local_ip_address = [(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in\
 [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]
        self.speech_file_path_for_misty = f'http://{local_ip_address}:{HTTP_SERVER_PORT}/robot_speech_files/speech.mp3'

        # Load the Google Gemini API key from the environment variable
        google_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
        if not google_api_key:
            raise ValueError("Please set the GOOGLE_GEMINI_API_KEY environment variable.")
        self.gemini_client = genai.Client(api_key=google_api_key)

        # get the system instruction prompt from a text file
        with open(llm_system_instruction_file) as f:
            system_instruction = f.read()
        f.close()

        # set up the generative text chat (new google-genai SDK)
        self.chat = self.gemini_client.chats.create(
            model='gemini-2.5-flash',
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type='application/json',
                system_instruction=system_instruction,
            )
        )

        # reset misty's LED and expression
        self.misty.change_led(100, 70, 160)
        self.misty.start_action(name="reset")

        # start the interactive text generation chat
        self.current_transcript = "Start conversation"
        self.execute_human_robot_dialogue()

    def execute_human_robot_dialogue(self):

        # keep running until you hit Ctrl+C or the genAI text model believes the conversation is done
        while True:

            # send the user input to the generative model and get the response
            user_input = self.current_transcript
            raw_response = self.chat.send_message(user_input)

            # process the response and extract the text for the robot to say and the expression for the robot to display
            response_json_dict = json.loads(raw_response.text)
            response_text = response_json_dict["msg"]
            response_expression = response_json_dict["expression"]
            print("AI (text):\t", response_text)
            print("AI (expression):", response_expression)

            # if the response is empty, assume the interaction is done and shut down the interaction
            if (len(response_text) <= 3):
                print("No response from LLM, assuming interaction is done and shutting down interaction.")
                break

            # OpenAI text-to-speech: generating speech and saving to a file
            with self.openai_client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts", #tts-1 may also be a good choice, as it was designed with low latency
                voice="alloy", # TODO: select a different voice for misty, see all voice options and play around with them at https://www.openai.fm/
                input=response_text,
                instructions="Speak with a calm and encouraging tone.",
            ) as response:
                response.stream_to_file(self.speech_file_path_local)

            # play the speech file on Misty
            self.misty.play_audio(self.speech_file_path_for_misty, volume=self.volume)

            # set the expression for the robot
            if (response_expression in custom_actions):
                self.misty.start_action(name=response_expression)
            else:
                print("Expression not found in custom actions. Using default expression.")
                self.misty.start_action(name="reset")

            # get the length of the audio file Misty is playing
            audio = MP3(self.speech_file_path_local)
            audio_info = audio.info
            audio_file_length = audio_info.length

            # wait for the audio file to finish playing before starting to listen again
            delay_for_stt = 2.0
            if (audio_file_length > delay_for_stt):
                # wait for the audio file to finish playing
                sleep(audio_file_length - delay_for_stt)

            # start listening again
            self.start_listening()


    def start_listening(self):
        # reset the robot's expression
        self.misty.start_action(name="reset")

        # change the LED to blue to indicate that the robot is listening
        self.misty.change_led(0, 199, 252)
        print("Rev.ai speech-to-text listening")

        # reset listening state
        self.current_transcript = ""
        self.listening_done = False

        # ensure the logs directory exists for saving user utterances
        date = datetime.now().strftime("%m-%d-%Y-%H-%M")
        logs_dir = './logs/' + date + '/'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        log_file_path = logs_dir + str(self.user_utterance_counter) + '.wav'
        self.user_utterance_counter += 1

        # STEP 1: open a microphone stream using PyAudio with a callback that
        # pushes audio chunks into a thread-safe buffer. The main thread
        # yields these chunks to Rev.ai without blocking the audio device.
        audio_buffer = Queue()
        raw_audio_bytes = []
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=AUDIO_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK,
            stream_callback=lambda in_data, fc, ti, sf: (
                audio_buffer.put(in_data), pyaudio.paContinue
            )[1]
        )
        stream.start_stream()

        # STEP 2: configure the Rev.ai streaming client for raw PCM audio.
        # MediaConfig takes positional args: (content_type, layout, rate, format, channels)
        media_config = MediaConfig('audio/x-raw', 'interleaved', AUDIO_RATE, 'S16LE', 1)
        rev_client = RevAiStreamingClient(self.revai_access_token, media_config)

        # STEP 3: define a generator that yields mic chunks to Rev.ai. The
        # generator stops when self.listening_done becomes True (set below
        # once we detect end-of-utterance).
        def audio_generator():
            while not self.listening_done:
                try:
                    chunk = audio_buffer.get(timeout=0.1)
                    raw_audio_bytes.append(chunk)
                    yield chunk
                except Exception:
                    continue

        # STEP 4: start streaming and process responses. Rev.ai sends two
        # kinds of messages we care about: 'partial' (interim hypotheses) and
        # 'final' (stable hypotheses). We accumulate the finals and treat a
        # gap of SILENCE_TIMEOUT seconds after a final as end-of-utterance.
        response_gen = rev_client.start(audio_generator())
        partial_texts = []
        final_texts = []
        silence_start = [None]

        try:
            for response in response_gen:
                response_dict = json.loads(response) if isinstance(response, str) else response
                msg_type = response_dict.get('type')
                elements = response_dict.get('elements', [])

                if msg_type == 'partial':
                    text = ' '.join(el['value'] for el in elements if el.get('type') == 'text')
                    if text:
                        partial_texts.append(text)
                        silence_start[0] = None   # reset silence timer on new speech
                elif msg_type == 'final':
                    text = ''.join(el['value'] for el in elements if el.get('type') == 'text')
                    if text.strip():
                        final_texts.append(text.strip())
                        print(f"Final: {text.strip()}")
                        silence_start[0] = time.time()   # start silence countdown

                # end-of-utterance: silence timeout after at least one final
                if silence_start[0] is not None:
                    if time.time() - silence_start[0] >= SILENCE_TIMEOUT:
                        self.listening_done = True
                        break

        except Exception as e:
            print(f"Rev.ai error: {e}")
        finally:
            # STEP 5: finish the connection and stop the microphone stream
            rev_client.end()
            stream.stop_stream()
            stream.close()
            pa.terminate()

            # change the LED back to purple to indicate that the robot is not listening
            self.misty.change_led(100, 70, 160)

        # save the user utterance as a raw WAV file for logging
        try:
            import wave
            with wave.open(log_file_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)   # 16-bit = 2 bytes
                wf.setframerate(AUDIO_RATE)
                wf.writeframes(b''.join(raw_audio_bytes))
        except Exception as e:
            print(f"Could not save utterance log: {e}")

        # concatenate the final hypotheses into the full utterance
        utterance = " ".join(final_texts).strip()
        if not utterance:
            utterance = " ".join(partial_texts).strip()
        print(f"Speech Final: {utterance}")

        # saves the utterance to the current_transcript variable
        self.current_transcript = utterance if utterance else "[silence]"

        print("Rev.ai STT stopped")


if __name__ == "__main__":

    # get Misty IP address
    if len(sys.argv) != 2:
        print("Usage: python llm_based_human_robot_dialogue_revai.py <Misty's IP Address>")
        sys.exit(1)
    misty_ip_address = sys.argv[1]

    # set up the MistyRobot object
    # TODO: modify the system instruction text file to allow the robot to execute the
    #       "Three Good Things" exercise
    misty_robot = MistyRobot(misty_ip_address, 'three_good_things_system_instruction.txt')
