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
elecHardware = False
photoHardware = False


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

##figPathR = 'C:\Python27\shape-detection\shape-detection\snapshots\R'
##figPathB = 'C:\Python27\shape-detection\shape-detection\snapshots\B'
##figSavePath = 'C:\Python27\shape-detection\shape-detection\Match'
##
##capTemp=r'C:\Python27\shape-detection\shape-detection\templates\caps'
##icTemp=r'C:\Python27\shape-detection\shape-detection\templates\ic'
##ledTemp=r'C:\Python27\shape-detection\shape-detection\templates\led'
##resTemp=r'C:\Python27\shape-detection\shape-detection\templates\resistors'
##
##numRails=16

#---end initialize---

#---run loop---
    #TBD
print "Initialization complete...."
    
    #wait for user input - button press
    #if button pressed was "check component":
        #LINDSAYS MODULE(s) activated - take picture, analyse picture
text = raw_input("Calibrate camera? (hit enter to continue)")

LETSCALIBRATE = True #NOTE: everytime a new component goes in - we're going to calibrate
while(LETSCALIBRATE):
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_AUTOFOCUS,1)
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
    calibrate(frame) #NOTE: 11/01/19 - MAY NEED TO UPDATE CALIBRATE FUNC
    cv2.destroyAllWindows()
    vc.release()

    moveOn = raw_input("Calibration done...move on to Snapshot? (yes/no): ") 
    if (moveOn == "yes"):
        LETSCALIBRATE = False
    else:
        continue


LETSBUILD = True
while(LETSBUILD):

    LINDSAYISGO = True
    while(LINDSAYISGO):
        
        #snapshot
        ###initiate video feed after the calibration is finished
        cv2.namedWindow("backgroundSubtract")#defining a window
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0) #start and define video capture
        fgbg=cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=75, detectShadows=False) #setting function parameters, not actual frames
        subFlag=1 #later
        if vc.isOpened(): # try to get the first frame - check if camera is opened
            rval, frame = vc.read()   #read the frame for fun!
            fgmask = fgbg.apply(frame) #apply the subtract for fun!
        else:
            rval = False #if it doesn't work, we have problems....deal with those later!
        while rval:
                    cv2.imshow("preview", frame) #show preview window - frame goes here
                    cv2.imshow("backgroundSubtract",fgmask) #show background sub window - fgmask goes here
                    rval, frame = vc.read() #take another frame - for fun!
                    if subFlag==1: #subFlag means we want to background subtract
                        fgmask = fgbg.apply(frame) #background subtract from frame
                    key = cv2.waitKey(20) #check for userkey every whatever # of frames?    
                    if key == 27: # exit on ESC - get out of jail free
                        cv2.destroyAllWindows()
                        vc.release()
                        break
                    if key == 98: #pause the videoCapture for user to place component
                        subFlag=0 #keep taking video feed but without any background subtract - in future may stop video here then background subtract after
                    if key == 99: #component has been placed. Now time to detect and locate
                        #vc.release() 
                        subFlag=1 
                        #wait 3 seconds for the camera to focus
                        time.sleep(3)
                        rval, frame = vc.read() #read the frame
                        fgmask = fgbg.apply(frame) #subtract
                        now=datetime.datetime.now() #getting time for filename
                        # save the real frame and background subtracted frame
                        fileName = 'fig' + now.strftime("%Y-%m-%d %H%M%S") 
                        completeNameB = os.path.join(figPathB, fileName + 'B.png')
                        completeNameR = os.path.join(figPathR, fileName + 'R.png')          
                        snapshot(completeNameB,rval,fgmask) #save snapshot
                        snapshot(completeNameR,rval,frame)
                        #component detection
                        compType=componentDetect(frame, resTemp, capTemp,icTemp,ledTemp)
                        #component location
                        railOne,railTwo=componentLocate(fgmask,frame, resCount,capCount,ledCount,icCount,numRails)
                        # continue while loop for user to add another component.
                        
        cv2.destroyAllWindows()
        

            #send packaged output from LINDSAYS MODULE(s) to JAMESONS MODULE(s)
            #JAMESONS MODULE(s) activated - communication with arduino and subsequent analysis of readings
       

        moveOn = raw_input("Lindsays modules done...move on to Jamesons? (yes/no): ")
        if (moveOn == "yes"):
            LINDSAYISGO = False
            print "Moving on to Jamesons mods...."
        else:
            continue
        
##    JAMESONISGO = True
##    while(JAMESONISGO):
##
##        lastComponentName = compType #always resistor for now
##        railOne = int(railOne) - 1
##        railTwo = int(railTwo) - 1
##        
##        if (text == "kill"):
##            break
##        else:
##            contents = text.split()
##            if lastComponentName == "resistor":
##                otherComponents = [] #FOR TIME BEING - ASSUME NO OTHER COMPONENTS IN PARALLEL
##                # COMMAND: buildResistor(self, rail1, rail2, otherComponents=[])
##                # Checks the resistance of a resistor between rail1 and rail2, and returns the resistance value in Ohms
##                #
##                # eg. arduino.buildResistor(0, 2, [["resistor", 1000], ["resistor", 5000]])
##                #  Note that if you don't have any other components in parallel then you only have to provide 2 arguments
##                #   since otherComponents will default to [] if nothing is provided
##                print arduino.buildResistor(int(railOne), int(railTwo), otherComponents) #NOTE: railOne and railTwo
##            elif (lastComponentName == "capacitor"):
##                otherComponents = []
##                # COMMAND: buildCapacitor(self, rail1, rail2, otherComponents=[])
##                # Checks the capacitance of a capacitor between rail1 and rail2, and returns the capacitance value in uF
##                # eg. arduino.buildCapacitor(0, 1)
##                print arduino.buildCapacitor(int(railOne), int(railTwo), otherComponents)
##            elif (lastComponentName == "diode"):
##                otherComponents = []
##                # COMMAND: buildDiode(self, rail1, rail2, otherComponents=[])
##                # Checks the polarity of the diode between rail1 and rail2, and returns either arduinoController.CONSTANTS["FORWARD_BIAS"]
##                #  if rail1 is the positive end, or arduinoController.CONSTANTS["REVERSE_BIAS"] if rail2 is the positive end
##                # eg. arduino.buildDiode(5, 6)
##                val = arduino.buildDiode(int(railOne), int(railTwo), otherComponents)
##                if val == arduinoController.CONSTANTS["FORWARD_BIAS"]:
##                    print "Forward Biased. Flows from rail " + railOne + " to " + railTwo
##                else:
##                    print "Reverse Biased. Flows from rail " + railOne + " to " + railTwo
##            else:
##                print "Error: Unexpected input. Discarding command"
##
##            moveOn = raw_input("Jamesons modules done...move on to Abis? (yes/no): ")
##            if (moveOn == "yes"):
##                JAMESONISGO = False
##                print "Moving on to Abis mods...."
##            else:
##                continue
##                
##
####    nextComponent = Component()    
####    nextComponent.nextComponent("R","9.95k","default",2,4,[0,0,0,0]) #VALUES NEED TO BE CORRECTED
####    grid.addComponent(nextComponent)
####
####    grid.drawGrid()
##
##    moveOn = raw_input("Component added...move on to Check? (yes/no): ")
##    if (moveOn == "yes"):
##        LETSBUILD = False
##        print "Moving on to CHECK mode...."
##    else:
##        continue
##
##
##LETSCHECK = True
##while(LETSCHECK):
##    print "Here will be checking code.....TBD"
##    moveOn = raw_input("Board has been checked...power on board? (yes/no): ")
##    if (moveOn=="yes"):
##        print "Powering on!"
##        LETSCHECK = False #move on for now...
##    else:
##        continue
##
### This one switches the Arduino to the Run mode
##arduino.changeMode(arduinoController.CONSTANTS["MODE_RUN"])
### This one actually turns the power on and off
##arduino.runPower(arduinoController.CONSTANTS["SWITCH_ON"])
##
##LETSRUN = True
##count = 0
##while(LETSRUN):   
##    railNums = []
##        for i in contents[1:]:
##            if (not arduinoController.isInt(i)):
##                print "Invalid input"
##                continue
##            else:
##                railNums.append(int(i))
##        ans = arduino.runVoltage(railNums)
##        for i in ans:
##            print i, ans[i]
##
##        count = count + 1
##        if (count > 100):
##            moveOn = raw_input("Board has been checked...power on board? (yes/no): ")
##                if (moveOn=="yes"):
##                print "Powering on!"
##                LETSCHECK = False #move on for now...
##        else:
##            continue

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



