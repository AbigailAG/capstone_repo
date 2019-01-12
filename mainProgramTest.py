### HOW TO USE THE arduinoController.py LIBRARY
# Note: This file is an example of how to use the arduinoController library and how to interface with it given command line input
# Step 1: Copy all the files from Github and dump them into the same directory as the file you're executing them in

# Step 2: Import the arduinoController library and the ArduinoInterface class as follows below
# Note that the library uses other libraries that I'm not sure you have by default (particularly serial and numpy),
#  so if those aren't installed you need to install them.
import arduinoController
from arduinoController import ArduinoInterface



def main():

    # Step 3: Instantiate the ArduinoInterface object
    #  This will find the Arduino if it's hooked up to the computer, verify serial communications,
    #   and run an initialization sequence to test some of the electrical properties of the Arduino and its circuit.
    #  Note that it's wrapped around a try-except statement. If an error is raised here, then the Arduino did not properly
    #   hook up with the computer and you probably need to prompt the user to reconnect the Arduino and then reinstantiate the object
    arduino = None
    try:
        arduino = ArduinoInterface()
    except AssertionError as error:
        print error
        print "Exiting as a result"
        exit()

    # Step 4: Call the methods available in the ArduinoInterface object based on your needs
    # NOTE: Some things you should know about the ArduinoInterface class to get how it works
    #  The Arduino itself has 4 different states, each of which have different commands:
    #  1. MODE_INIT: This is the initialized mode. The Arduino starts in this mode when booted up, or whenever serial
    #                communication is reestablished between it and the computer. In this mode, all the serial verification
    #                and initialization sequence is run. Upon instantiation, the Arduino enters this mode, then immediately
    #                runs its tests as then jumps into MODE_BUILD once it's done. So you should realistically never have
    #                a scenario where you run commands in this mode
    #  2. MODE_BUILD: This is the mode the Arduino should be in while the circuit is being built, and you need to run a
    #                 component characterization to check its values. The Arduino is in this state as soon as the object
    #                 is instantiated.
    #                 Available methods you can run: changeMode, buildResistor, buildCapacitor, buildDiode
    #  3. MODE_RUN: This is the mode the Arduino should be in while the breadboard circuit is being turned on, and you want
    #               to turn the power on or take measurements
    #               Available methods you can run: changeMode, runPower, runVoltage, runCurrent
    #  4. MODE_DEBUG: This is a user mode you can jump into if you need to assume direct control over the ArduinoInterface
    #                 object and the Arduino itself. You should not go into here at all during normal operation. But this
    #                 is effectively a backdoor to run any of the individual subroutines programmed onto the Arduino
    #                 Available methods you can run: changeMode, debugCommand
    #  You jump between different modes by using the changeMode method. Note that you should use the arduinoController.CONSTANTS
    #   variable to pull in some of the required constants to ensure that you and the ArduinoInterface object are 'speaking
    #   the same language'.
    #  More details are given in the comments below, and in the arduinoController.py file

    # Future improvements that can be made:
    #  Backup methods for checking if Lindsay's reported rails are off by an adjacent rail
    #  Sensing methods (ie. accessors) so you can check the current state of something in the object
    #  ?? Let me know if you need anything else added

    # This is an infinite loop to just receive any commands that you have.
    # In here contains pretty much all the commands you'll need to operate and interface with the ArduinoInterface.
    # All commands you can run will be prefaced with a '# COMMAND' comment, and some information about them
    # So check here for an example of how to use the methods, and then refer to the arduinoController.py file to
    #  see the required details for inputs, processing, and outputs of each method.
    while (True):
        #
        text = raw_input("Please enter a request: ")
        print(text)
        if (text == "kill"):
            break
        else:
            contents = text.split()
            if (len(contents) == 2) and (contents[0] == "mode"):
                # COMMAND: changeMode(self, mode)
                # Use this to change between MODE_BUILD and MODE_RUN in normal operation, or MODE_DEBUG if something
                #  goes terribly wrong and you need operator intervention
                # eg. arduino.changeMode(arduinoController.CONTENTS["MODE_BUILD"])
                print arduino.changeMode(int(contents[1]))
            elif (len(contents) == 2) and (contents[0] == "change"):
                # Don't use this in normal circumstances
                print arduino.changeReferenceResistor(int(contents[1]))
            elif (len(contents) >= 3) and (contents[0] == "resistor"):
                # COMMAND: buildResistor(self, rail1, rail2, otherComponents=[])
                # Checks the resistance of a resistor between rail1 and rail2, and returns the resistance value in Ohms
                #
                # eg. arduino.buildResistor(0, 2, [["resistor", 1000], ["resistor", 5000]])
                #  Note that if you don't have any other components in parallel then you only have to provide 2 arguments
                #   since otherComponents will default to [] if nothing is provided
                otherComponents = []
                if (len(contents) > 3):
                    for i in contents[3:]:
                        split = i.split(",")
                        if arduinoController.isFloat(split[1]):
                            otherComponents.append([split[0], float(split[1])])
                        else:
                            otherComponents.append(split)
                print arduino.buildResistor(int(contents[1]), int(contents[2]), otherComponents)
            elif (len(contents) >= 3) and (contents[0] == "capacitor"):
                # COMMAND: buildCapacitor(self, rail1, rail2, otherComponents=[])
                # Checks the capacitance of a capacitor between rail1 and rail2, and returns the capacitance value in uF
                # eg. arduino.buildCapacitor(0, 1)
                otherComponents = []
                if (len(contents) > 3):
                    for i in contents[3:]:
                        split = i.split(",")
                        if arduinoController.isFloat(split[1]):
                            otherComponents.append([split[0], float(split[1])])
                        else:
                            otherComponents.append(split)
                print arduino.buildCapacitor(int(contents[1]), int(contents[2]), otherComponents)
            elif (len(contents) >= 3) and (contents[0] == "capacitor2"):
                # Obsoleted. Do not use
                otherComponents = []
                if (len(contents) > 3):
                    for i in contents[3:]:
                        split = i.split(",")
                        if arduinoController.isFloat(split[1]):
                            otherComponents.append([split[0], float(split[1])])
                        else:
                            otherComponents.append(split)
                print arduino.buildCapacitor2(int(contents[1]), int(contents[2]), otherComponents)
            elif (len(contents) >= 3) and (contents[0] == "diode"):
                # COMMAND: buildDiode(self, rail1, rail2, otherComponents=[])
                # Checks the polarity of the diode between rail1 and rail2, and returns either arduinoController.CONSTANTS["FORWARD_BIAS"]
                #  if rail1 is the positive end, or arduinoController.CONSTANTS["REVERSE_BIAS"] if rail2 is the positive end
                # eg. arduino.buildDiode(5, 6)
                otherComponents = []
                if (len(contents) > 3):
                    for i in contents[3:]:
                        split = i.split(",")
                        if arduinoController.isFloat(split[1]):
                            otherComponents.append([split[0], float(split[1])])
                        else:
                            otherComponents.append(split)
                val = arduino.buildDiode(int(contents[1]), int(contents[2]), otherComponents)
                if val == arduinoController.CONSTANTS["FORWARD_BIAS"]:
                    print "Forward Biased. Flows from rail " + contents[1] + " to " + contents[2]
                else:
                    print "Reverse Biased. Flows from rail " + contents[2] + " to " + contents[1]
            elif (len(contents) >= 3) and (contents[0] == "resistorComp"):
                # Obsoleted. Do not use
                otherComponents = []
                if (len(contents) > 3):
                    for i in contents[3:]:
                        split = i.split(",")
                        if arduinoController.isFloat(split[1]):
                            otherComponents.append([split[0], float(split[1])])
                        else:
                            otherComponents.append(split)
                print arduino.buildResistorComp(int(contents[1]), int(contents[2]), otherComponents)
            elif (len(contents) >= 3) and (contents[0] == "capacitorComp"):
                # Internal testing. Do not use
                otherComponents = []
                if (len(contents) > 3):
                    for i in contents[3:]:
                        split = i.split(",")
                        if arduinoController.isFloat(split[1]):
                            otherComponents.append([split[0], float(split[1])])
                        else:
                            otherComponents.append(split)
                print arduino.buildCapacitorComp(int(contents[1]), int(contents[2]), otherComponents)
            elif (len(contents) >= 3) and (contents[0] == "capacitorMix"):
                # Obsoleted. Do not use
                otherComponents = []
                if (len(contents) > 3):
                    for i in contents[3:]:
                        split = i.split(",")
                        if arduinoController.isFloat(split[1]):
                            otherComponents.append([split[0], float(split[1])])
                        else:
                            otherComponents.append(split)
                print arduino.buildCapacitorMix(int(contents[1]), int(contents[2]), otherComponents)
            elif (len(contents) >= 2) and (contents[0] == "power"):
                # COMMAND: runPower(self, option)
                # Either turns power to the circuit on, off, or the opposite of whatever state it was in before. Give commands
                #  to this method by referencing the arduinoController.CONSTANTS for the switch terms (seen in this example)
                # eg. arduino.runPower(arduinoController.CONSTANTS["SWITCH_ON"])
                if (contents[1] == "on"):
                    arduino.runPower(arduinoController.CONSTANTS["SWITCH_ON"])
                elif (contents[1] == "off"):
                    arduino.runPower(arduinoController.CONSTANTS["SWITCH_OFF"])
                elif (contents[1] == "toggle"):
                    arduino.runPower(arduinoController.CONSTANTS["SWITCH_TOGGLE"])
            elif (len(contents) >= 2) and (contents[0] == "voltage"):
                # COMMAND: runVoltage(self, rails)
                # Reads the voltages of all the rails specified in the list provided, and returns a dictionary of values,
                #  with the rail number being the key and the voltage on that rail being the key value
                # eg. arduino.runVoltage([0, 1, 2, 5, 6, 7]) -> {[0, 0], [1, 5], [2, 3.3], [5, 4.2], [6, 2.8], [7, 1.6]}
                railNums = []
                for i in contents[1:]:
                    if (not arduinoController.isInt(i)):
                        print "Invalid input"
                        continue
                    else:
                        railNums.append(int(i))
                ans = arduino.runVoltage(railNums)
                for i in ans:
                    print i, ans[i]
            elif (all(arduinoController.isInt(x) for x in contents)):
                # Need to do more extensive test here in case inputs are not ints, but for now this can do
                # COMMAND: debugCommand(self, bytes)
                # Interface directly with the Arduino serial communications and send a series of bytes through the Arduino
                #  serial communications.
                # Note that you really need to make sure that you know what you're sending when you use this, as it is a
                #  direct communication port to the Arduino. Also you probably want to make use of the arduinoController.CONSTANTS
                #  dictionary to make communications easier.
                # eg. arduino.debugCommand([5, 1])
                if arduinoController.DEBUG: print "Hello let's debug"
                # The arduino.debugSerial method is really a wrapper method around debugCommand to convert a string input
                #  into a list of ints first.
                print arduino.debugSerial(text)
            elif (text.split(":")[0] == "eval"):
                # This is direct code injection. Don't use.
                eval(text.split(":")[1:])
            else:
                print "Error: Unexpected input. Discarding command"


# Call main function
if __name__ == '__main__':
    main()
