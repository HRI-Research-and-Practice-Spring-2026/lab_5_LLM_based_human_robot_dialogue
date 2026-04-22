import os, time, sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), 'Python-SDK'))
from mistyPy.Robot import Robot
from mistyPy.Events import Events


# TODO: add 5 more custom actions for the robot
custom_actions = {
    "reset": "IMAGE:e_DefaultContent.jpg; ARMS:40,40,1000; HEAD:-5,0,0,1000;",
    "head-up-down-nod": "IMAGE:e_DefaultContent.jpg; HEAD:-15,0,0,500; PAUSE:500; HEAD:5,0,0,500; PAUSE:500; HEAD:-15,0,0,500; PAUSE:500; HEAD:5,0,0,500; PAUSE:500; HEAD:-5,0,0,500; PAUSE:500;",
    "hi": "IMAGE:e_Admiration.jpg; ARMS:-80,40,100;",
    "listen": "IMAGE:e_Surprise.jpg; HEAD:-6,30,0,1000; PAUSE:2500; HEAD:-5,0,0,500; IMAGE:e_DefaultContent.jpg;"
}

if __name__ == "__main__":

    # get Misty IP address
    if len(sys.argv) != 2:
        print("Usage: python misty_introduction.py <Misty's IP Address>")
        sys.exit(1)
    misty_ip_address = sys.argv[1]

    # set up the MistyRobot object 
    misty_robot = Robot(misty_ip_address)
    
    # create all of our custom actions
    for action_name, action_script in custom_actions.items():
        misty_robot.create_action(
            name = action_name,
            script = action_script,
            overwrite = True
        )

    # execute the action you want to test
    misty_robot.start_action(name="hi")

    # after waiting 2 seconds, put the robot back to its reset expression
    time.sleep(2)
    misty_robot.start_action(name="reset")