#--------------------------------------
#SMARTCIRCUIT Main Control Module
#
# required text files:
#
#
#--------------------------------------

#imports
from graphics import *
from Component import *
from GridAndNodesAndComps import *

import arduinoController
from arduinoController import ArduinoInterface

from cam_mod import *


#TBD
#end imports

#---initialize---
print "initializing program"
    #TBD
    #setup GUI
#RUN WITH ARDUINO WORKING?
elecHardware = True
photoHardware = True


    #turn on arduino - wait for input mode on arduino

if (elecHardware):
    print "Electrical characterization hardware connected!"
    try:
        arduino = ArduinoInterface()
    except AssertionError as error:
        print error
        print "Exiting as a result - Arduino Interface failed"
        exit()
else:
    print "Electrical characterization hardware NOT connected - test mode"

#setup holding variable
grid = GridAndNodesAndComps()
grid.initialize()

figPathR = 'cam_data\snapshots\R'
figPathB = 'cam_data\snapshots\B'
figSavePath = 'cam_data\Match'
##
capTemp=r'cam_data\templates\caps'
icTemp=r'cam_data\templates\ic'
ledTemp=r'cam_data\templates\led'
resTemp=r'cam_data\templates\resistors\R'
my_path = os.path.abspath(os.path.dirname(__file__))
##


#---end initialize---

#---run loop---
    #TBD
print "Initialization complete...."

    #wait for user input - button press
    #if button pressed was "check component":
        #LINDSAYS MODULE(s) activated - take picture, analyse picture
text = raw_input("Calibrate camera? (y/n):")
if (text == "y"):
    LETSCALIBRATE = True
else:
    LETSCALIBRATE = False

#LETSCALIBRATE = True #NOTE: everytime a new component goes in - we're going to calibrate
while(LETSCALIBRATE):
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    vc.set(cv2.CAP_PROP_AUTO_EXPOSURE,1)
    vc.set(cv2.CAP_PROP_AUTOFOCUS,1)
    vc.set(cv2.CAP_PROP_BRIGHTNESS,145)
    vc.set(cv2.CAP_PROP_CONTRAST,35)
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    while rval: #switch to for loop for 5 seconds
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            cv2.destroyAllWindows()
            vc.release()
            break
        if key == 99:
            rval, frame = vc.read()
            break
    calibrate(frame)
    cv2.destroyAllWindows()
    vc.release()

    moveOn = raw_input("Calibration done...move on to Snapshot? (y/n): ")
    if (moveOn == "y"):
        LETSCALIBRATE = False
    else:
        continue


LETSBUILD = True
while(LETSBUILD):

    LINDSAYISGO = True
    while(LINDSAYISGO):
        #load bb points
        bbpoints=pickle.load(open("cam_data/bbpoints.obj","rb"))
        #rect_r=np.array(pickle.load(open("cam_data/rect_r.obj","rb"))).flatten()
        r2_rails=pickle.load(open("cam_data/r2_rails.obj","rb"))
        r3_rails=pickle.load(open("cam_data/r3_rails.obj","rb"))
        #snapshot
        ###initiate video feed after the calibration is finished
        cv2.namedWindow("backgroundSubtract")#defining a window
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0) #start and define video capture
        vc.set(cv2.CAP_PROP_AUTO_EXPOSURE,1)
        vc.set(cv2.CAP_PROP_AUTOFOCUS,1)
        vc.set(cv2.CAP_PROP_BRIGHTNESS,150)
        vc.set(cv2.CAP_PROP_CONTRAST,45)
        fgbg=cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=79, detectShadows=True) #setting function parameters, not actual frames
        subFlag=1 #later
        if vc.isOpened(): # try to get the first frame - check if camera is opened
            rval, frameRaw = vc.read()
            print "waiting ....",
            for i in xrange(1,500):
                print ".",
                rval, frameRaw = vc.read()
                frame=four_point_transform(frameRaw,bbpoints)
                fgmask = fgbg.apply(frame)
                cv2.waitKey(20)
                cv2.imshow("backgroundSubtract",fgmask)
                cv2.imshow("preview",frame)
        else:
            rval = False #if it doesn't work, we have problems....deal with those later!
        while rval:
            #perform transformations
            ## four point transform image
            rval, frameRaw= vc.read()
            
            frame=four_point_transform(frameRaw,bbpoints)
            railOver=frame.copy()
            for ind in xrange(len(r2_rails)):
                rect_r2=np.array([r2_rails[ind]],dtype="int32")
                cv2.drawContours(railOver,rect_r2,-1,(200,25,0),1)
                rect_r3=np.array([r3_rails[ind]],dtype="int32")
                cv2.drawContours(railOver,rect_r3,-1,(200,25,0),1)
            cv2.imshow("preview", railOver)
            if subFlag==1: #subFlag means we want to get video feed
                fgmask = fgbg.apply(frame) #background subtract from frame
                cv2.imshow("backgroundSubtract",fgmask)

            key = cv2.waitKey(20) 
            if key == 27: # exit on ESC - get out of jail free
                cv2.destroyAllWindows()
                vc.release()
                break
            if key == 98: #pause the videoCapture for user to place component
                subFlag=0 #stop taking video feed background subtract - in future may stop video here then background subtract after
            if key == 99: #component has been placed. Now time to detect and locate
                subFlag=1
                #wait 3 seconds for the camera to focus
                print "waiting ....",                
                for i in xrange(1,200):
                    rval, frameRaw = vc.read() #read the frame
                    key = cv2.waitKey(20)
                    print ".",
                    frame=four_point_transform(frameRaw,bbpoints)                    
                    cv2.imshow("preview",frame)                
                cv2.imshow("backgroundSubtract",fgmask)
                for i in xrange(1,15):
                    rval, frameRaw = vc.read() #read the frame
                    key = cv2.waitKey(20)
                    print ".",
                    frame=four_point_transform(frameRaw,bbpoints)
                    fgmask = fgbg.apply(frame) #subtract
                    cv2.imshow("preview",frame)
                    cv2.imshow("backgroundSubtract",fgmask)

                
                now=datetime.datetime.now() #getting time for filename
                # save the real frame and background subtracted frame
                my_path = os.path.abspath(os.path.dirname(__file__))
                fileName = 'fig' + now.strftime("%Y-%m-%d %H%M%S")
                #completeNameB = os.path.join(figPathB, fileName + 'B.png')
                completeNameB=figPathB+ '\\' + fileName+'B.png'
                #completeNameR = os.path.join(figPathR, fileName + 'R.png')
                completeNameR=figPathR+'\\'+fileName+'R.png'
                snapshot(completeNameB,rval,fgmask) #save snapshot
                snapshot(completeNameR,rval,frame)
                #component detection
                compType=componentDetect(frame, fgmask, resTemp, capTemp,icTemp,ledTemp,my_path)
                #component location
                railOne,railTwo=componentLocate(fgmask,frame, 1,0,0,0)
                print railOne,railTwo
                # continue while loop for user to add another component.        
        cv2.destroyAllWindows()


            #send packaged output from LINDSAYS MODULE(s) to JAMESONS MODULE(s)
            #JAMESONS MODULE(s) activated - communication with arduino and subsequent analysis of readings


        moveOn = raw_input("Lindsays modules done...move on to Jamesons? (yes/no): ")
        if (moveOn == "yes"):
            LINDSAYISGO = False
            vc.release()
            manualType = raw_input("Confirm type? (y/n) ")
            if (manualType == "y"):
                print "Moving on to Jamesons mods...."
            else:
                compType = raw_input("enter comp type: ")
        else:
            continue

    if int(railOne)>=31: ##then it is in region 2 (rails 31-60)
        railOne=int(railOne)-23 #convert between 9-15
    else:
        railOne=int(railOne)-1 ##convert between 0-7
    if int(railTwo)<=30:        ## then it is in region 1 (rails 1-30)
        railTwo=int(railTwo)-1 ## convert between 0-7
    else:
        railTwo=int(railTwo)-23  ## convert between 9-15
    print "railOne Converted", railOne
    print "railTwo Converted", railTwo
    #manual input
    
    
    JAMESONISGO = True
    while(JAMESONISGO):

        #railOne = raw_input("Rail1: ")
        #railOne = int(railOne)
        #railTwo = raw_input("Rail2: ")
        #railTwo = int(railTwo)
        #compType = raw_input("Comp type: ")

        lastComponentName = compType #always resistor for now
        #changing value to 0-15
        
        
        

        testRailCheck = grid.checkDummy(railOne,railTwo)

        if (text == "kill"):
            break
        else:
            contents = text.split()
            if lastComponentName == "resistor":
                otherComponents = [] #FOR TIME BEING - ASSUME NO OTHER COMPONENTS IN PARALLEL
                # COMMAND: buildResistor(self, rail1, rail2, otherComponents=[])
                # Checks the resistance of a resistor between rail1 and rail2, and returns the resistance value in Ohms
                #
                # eg. arduino.buildResistor(0, 2, [["resistor", 1000], ["resistor", 5000]])
                #  Note that if you don't have any other components in parallel then you only have to provide 2 arguments
                #   since otherComponents will default to [] if nothing is provided
                compVal= arduino.buildResistor(int(railOne), int(railTwo), testRailCheck) #NOTE: railOne and railTwo
            elif (lastComponentName == "capacitor"):
                otherComponents = []
                # COMMAND: buildCapacitor(self, rail1, rail2, otherComponents=[])
                # Checks the capacitance of a capacitor between rail1 and rail2, and returns the capacitance value in uF
                # eg. arduino.buildCapacitor(0, 1)
                compVal = arduino.buildCapacitor(int(railOne), int(railTwo), otherComponents)
                print "CAP MEASURE"
                print compVal
                compVal = compVal*1000 #reporting in nF for time being
                
            elif (lastComponentName == "diode"):
                otherComponents = []
                # COMMAND: buildDiode(self, rail1, rail2, otherComponents=[])
                # Checks the polarity of the diode between rail1 and rail2, and returns either arduinoController.CONSTANTS["FORWARD_BIAS"]
                #  if rail1 is the positive end, or arduinoController.CONSTANTS["REVERSE_BIAS"] if rail2 is the positive end
                # eg. arduino.buildDiode(5, 6)
                compVal = arduino.buildDiode(int(railOne), int(railTwo), otherComponents)
                print "WHAT IS THIS"
                print compVal
                if compVal == arduinoController.CONSTANTS["FORWARD_BIAS"]:
                    print "Forward Biased. Flows from rail " + str(railOne) + " to " + str(railTwo)
                else:
                    print "Reverse Biased. Flows from rail " + str(railOne) + " to " + str(railTwo)
            else:
                print "Error: Unexpected input. Discarding command"
            print compVal

            print "Component is added!"
            nextComponent = Component()
            print "HERE!!! " + str(compType)
            nextComponent.newComponent(compType,str(int(compVal)),"default",railOne,railTwo) #VALUES NEED TO BE CORRECTED
            grid.addComponent(nextComponent)
            print grid.getNodes()
            grid.drawGrid()

            moveOn = raw_input("Jamesons modules done...move on to Abis? (yes/no): ")
            if (moveOn == "yes"):
                JAMESONISGO = False
                print "Moving on to Abis mods...."
            else:
                continue


##    nextComponent = Component()
##    nextComponent.newComponent("R",str(compVal),"default",railOne,railTwo,[0,0,0,0]) #VALUES NEED TO BE CORRECTED
##    grid.addComponent(nextComponent)

##    grid.drawGrid()

    moveOn = raw_input("Component added...move on to Check? (yes/no): ")
    if (moveOn == "yes"):
        LETSBUILD = False
        print "Moving on to CHECK mode...."
    else:
        continue


LETSCHECK = True
while(LETSCHECK):
    print "Here will be checking code.....TBD"
    moveOn = raw_input("Board has been checked...power on board? (yes/no): ")
    if (moveOn=="yes"):
        print "Powering on!"
        LETSCHECK = False #move on for now...
    else:
        continue

# This one switches the Arduino to the Run mode
arduino.changeMode(arduinoController.CONSTANTS["MODE_RUN"])
# This one actually turns the power on and off
arduino.runPower(arduinoController.CONSTANTS["SWITCH_ON"])

LETSRUN = True
count = 0
while(LETSRUN):
##    railNums = []
##    for i in [railOne,railTwo]:
##        if (not arduinoController.isInt(i)):
##            print "Invalid input"
##            continue
##        else:
##            railNums.append(int(i))
    railsToMeasure = grid.getRealNodes()
    ans = arduino.runVoltage(railsToMeasure)
    for i in ans:
        print i, ans[i]

    count = count + 1
    if (count > 100):
        moveOn = raw_input("Run board again? (yes/no): ")
        if (moveOn=="no"):
            print "Powering off!"
            LETSRUN = False #move on for now...
        else:
            continue

print "Turning off the board....."
arduino.runPower(arduinoController.CONSTANTS["SWITCH_OFF"])
arduino.changeMode(arduinoController.CONSTANTS["MODE_BUILD"])


        #JAMESONS MODULE(s) returns -> type of component, value of component, location of component
        #check output (will need to discuss who will be covering error handling....)
        #if data is good:
            #this data is saved to CURRENT_ARDUINO_DATA
            #this data is packaged
            #continue
        #if data is not good:
            #identify problem and possible solution to user - MORE CONSIDERATION REQUIRED HERE
        #check that LINDSAY AND JAMESON(s) results match
        #if results match:
            #continue
        #if not:
            #identify problem and possible solution to user - MORE CONSIDERATION REQUIRED HERE

        #package combined JAMESON AND LINDAYS data in final form
        #output component info to user
        #add component info to model (UPDATE MODEL function) - input CURRENT_PIC_DATA and CURRENT_ARDUINO_DATA
        #DRAW CIRCUIT - send input to this separate function
        #DRAW CIRCUIT returns image
        #done check component, end block

    #if button pressed was "modify component":
        #call MODIFY_COMPONENT function (the following is what all will go inside)
        #provide user with list of components to choose from, wait for choice
        #provide user with properties to modify, wait for modifications
        #check modifications (make sense)
        #make temporary update of model, and of circuit diagram (but hold onto old)
        #display updated circuit model - double check with user this is what they want
        #I DON'T KNOW IF WE WANT TO RUN ANOTHER ARDUINO/CAMERA CHECK HERE TO DOUBLE CHECK USER INPUT? - will need to be further discussed
        #if user confirms:
            #overwrite model and circuit diagram with temporary, get rid of old
            #continue
        #if user says no:
            #go back to providing user with properties to modify and continue from there
        #done modify component, end block

    #if button pressed was "backstage":
        #open secondary window showing history of functions and current functions (command line style)

    #if button pressed was "circuit complete":
        #confirm with user that circuit is complete
        #if yes:
            #continue
        #else:
            #go back to waiting at main interface
        #call DESIGNCHECKS (the following is what happens inside that function
        #assign each component a numerical position in the circuit (from start/power = 1 to end/ground = n), parallel components are equal
        #assign each component a list of adjacent component(s)

##n = 5
###at the moment, assuming there are a total of 5 nodes
##c = 10
###at the moment, assuming max 10 components


##    initGrid = [[0 for eachRow in range(n)] for eachCol in range(n)]
##    initGrid[0][2] = "P" #setting power as P
##    initGrid[4][2] = "G"
##    initNodes = [[0 for eachRow2 in range(c)] for eachCol2 in range(n+1)]
##    return initGrid, initNodes
    #I'm thinking I'm going to want Grid and matrix classes, then keep track of variables that way.....TBD


##    firstnode = firstComp.getNodeI()
##    secondnode = firstComp.getNodeO()
##    nodes_firstcomp[0][firstnode] = 1
##    nodes_firstcomp[0][secondnode] = 1
##    compID = str(firstComp.getType())+str(0)
##    #lookUpTable[0] = [compID,str(firstComp.getType())+str(firstComp.getValue())]
##    nodes_firstcomp[0][-1] = str(firstComp.getType())+str(0)
##    gridmat_firstcomp[1][2] = str(firstComp.getType())+str(0)
##    counter = 1
##    return gridmat_firstcomp, nodes_firstcomp, counter#, lookUpTable

##    newn1 = secondComp.getNodeI()
##    newn2 = secondComp.getNodeO()
##    compID = str(secondComp.getType())+str(0)
##    nodes_secondComp[counter_secondcomp][newn1] = 1
##    nodes_secondComp[counter_secondcomp][newn2] = 1
##    nodes_secondComp[counter_secondcomp][-1] = compID
##    #lookUpTable_secondcomp[counter_secondcomp] = [compID,
##
##    THIScomp = nodes_secondComp[counter_secondcomp]
##    relationList = [0 for eachComp in range(counter_secondcomp-1)]
##    for allRows in range(c):
##        sumOfRows = 0
##        if (allRows == counter_secondcomp):
##            continue
##
##        for allCols in range(n):
##            sumOfRows = sumOfRows + nodes_secondComp[allRows][allCols] + THIScomp[allCols]
##
##            if (allCols == (n-1)):
##                relationList[allRows] = sumOfRows
##                if (sumOfRows == 2):
##                    #no relation between the two
##                elif (sumOfRows == 4):
##                    #parallel
##                elif (sumOfRows == 3):
##                    #series





    #if ((int(newn1) == int(nodes_secondcomp[0])) and (int(newn2) == int(nodes_secondcomp[1]))):
         #gridmat_secondcomp[1][1] = gridmat_secondcomp[1][2]
         #gridmat_secondcomp[1][2] = 0
         #gridmat_secondcomp[1][3] = str(secondComp.getType())+str(secondComp.getValue())
         #return gridmat_secondcomp,nodes_secondcomp
    #elif (int(newn1) == int(nodes_secondcomp[1])):
         #gridmat_secondcomp[2][2] = str(secondComp.getType())+str(secondComp.getValue())
         #return gridmat_secondcomp,nodes_secondcomp
    #elif (int(newn2) == int(nodes_secondcomp[0])):
         #gridmat_secondcomp[2][2] = gridmat_secondcomp[1][2]
         #gridmat_secondcomp[1][2] = str(secondComp.getType())+str(secondComp.getValue())
         #return gridmat_secondcomp,nodes_secondcomp


# Turn the power off
##arduino.runPower(arduinoController.CONSTANTS["SWITCH_OFF"])
