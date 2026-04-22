from google import genai
from google.genai import types
from openai import OpenAI
import pyaudio, wave, json, os, socket, sys, time
from dotenv import load_dotenv
from mutagen.mp3 import MP3
from datetime import datetime
from time import sleep
from queue import Queue

sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), 'Python-SDK'))
from mistyPy.Robot import Robot
from mistyPy.Events import Events


# HTTP server port — change this if port 8000 is busy on your machine
HTTP_SERVER_PORT = 8000

# PyAudio recording settings — used to capture mic input and send to Whisper
AUDIO_RATE = 16000
AUDIO_CHUNK = int(AUDIO_RATE / 10)       # 100ms chunks
SILENCE_THRESHOLD = 500                  # RMS amplitude below this = silence
SILENCE_DURATION = 2.0                   # seconds of silence before we stop recording
MAX_RECORDING_SECONDS = 30               # hard cap on any single utterance


# TODO: add 5 more custom actions for the robot
custom_actions = {
    "reset": "IMAGE:e_DefaultContent.jpg; ARMS:40,40,1000; HEAD:-5,0,0,1000;",
    "head-up-down-nod": "IMAGE:e_DefaultContent.jpg; HEAD:-15,0,0,500; PAUSE:500; HEAD:5,0,0,500; PAUSE:500; HEAD:-15,0,0,500; PAUSE:500; HEAD:5,0,0,500; PAUSE:500; HEAD:-5,0,0,500; PAUSE:500;",
    "hi": "IMAGE:e_Admiration.jpg; ARMS:-80,40,100;",
    "listen": "IMAGE:e_Surprise.jpg; HEAD:-6,30,0,1000; PAUSE:2500; HEAD:-5,0,0,500; IMAGE:e_DefaultContent.jpg;"
}


def compute_rms(frame_bytes):
    """Compute RMS amplitude of a 16-bit PCM audio frame."""
    import audioop
    return audioop.rms(frame_bytes, 2)   # 2 = sample width in bytes (16-bit)


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

        load_dotenv()

        # initialize the OpenAI client for both TTS (speech) and STT (Whisper)
        open_ai_api_key = os.getenv('OPEN_AI_API_KEY')
        if not open_ai_api_key:
            raise ValueError("Please set the OPEN_AI_API_KEY environment variable.")
        self.openai_client = OpenAI(api_key=open_ai_api_key)

        # variable that holds the latest transcript from Whisper
        self.current_transcript = ""
        self.user_utterance_counter = 1

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
        """
        Records audio from the local microphone using PyAudio, stops when it
        detects SILENCE_DURATION seconds of silence after hearing speech, then
        sends the recorded audio to OpenAI's Whisper API for transcription.
        """
        # reset the robot's expression
        self.misty.start_action(name="reset")

        # change the LED to blue to indicate that the robot is listening
        self.misty.change_led(0, 199, 252)
        print("OpenAI Whisper speech-to-text listening (record until silence)…")

        # ensure the logs directory exists for saving user utterances
        date = datetime.now().strftime("%m-%d-%Y-%H-%M")
        logs_dir = './logs/' + date + '/'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        utterance_wav_path = logs_dir + str(self.user_utterance_counter) + '.wav'
        self.user_utterance_counter += 1

        # STEP 1: open a microphone stream using PyAudio
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=AUDIO_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK
        )

        # STEP 2: record audio, stopping when we detect silence after speech.
        # Uses a simple volume-based voice activity detection: once we've
        # heard a chunk louder than SILENCE_THRESHOLD, we start checking for
        # silence; once SILENCE_DURATION seconds of silence pass, we stop.
        frames = []
        started_speaking = False
        last_sound_time = time.time()
        start_time = time.time()

        while True:
            try:
                chunk = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            except Exception as e:
                print(f"Audio read error: {e}")
                break
            frames.append(chunk)

            rms = compute_rms(chunk)
            now = time.time()

            if rms > SILENCE_THRESHOLD:
                if not started_speaking:
                    print("  (detected speech)")
                    started_speaking = True
                last_sound_time = now

            if started_speaking and (now - last_sound_time >= SILENCE_DURATION):
                print("  (detected end of speech)")
                break

            if now - start_time > MAX_RECORDING_SECONDS:
                print("  (hit max recording length)")
                break

        # STEP 3: close the microphone stream
        stream.stop_stream()
        stream.close()
        pa.terminate()

        # change the LED back to purple to indicate that the robot is not listening
        self.misty.change_led(100, 70, 160)

        # STEP 4: save the recording to disk as a WAV file (Whisper needs a file)
        with wave.open(utterance_wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)     # 16-bit = 2 bytes
            wf.setframerate(AUDIO_RATE)
            wf.writeframes(b''.join(frames))

        # STEP 5: if the user didn't speak, skip transcription
        if not started_speaking:
            print("No speech detected.")
            self.current_transcript = "[silence]"
            return

        # STEP 6: send the WAV file to OpenAI's Whisper API for transcription
        try:
            with open(utterance_wav_path, 'rb') as audio_file:
                transcription = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text",
                    language="en",
                )
            utterance = transcription.strip() if isinstance(transcription, str) else str(transcription).strip()
        except Exception as e:
            print(f"Whisper error: {e}")
            utterance = ""

        print(f"Speech Final: {utterance}")
        self.current_transcript = utterance if utterance else "[silence]"

        print("Whisper STT stopped")


if __name__ == "__main__":

    # get Misty IP address
    if len(sys.argv) != 2:
        print("Usage: python llm_based_human_robot_dialogue.py <Misty's IP Address>")
        sys.exit(1)
    misty_ip_address = sys.argv[1]

    # set up the MistyRobot object
    # TODO: modify the system instruction text file to allow the robot to execute the
    #       "Three Good Things" exercise
    misty_robot = MistyRobot(misty_ip_address, 'three_good_things_system_instruction.txt')
