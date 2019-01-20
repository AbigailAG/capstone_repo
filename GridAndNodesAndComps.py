#won't need random number generator in end
import random
import numpy as np
from Component import *
from graphics import *

class GridAndNodesAndComps:

    def __init__(self):
        self.n = 16 #total of 7 rows and columns in grid - MUST BE ODD   
        self.c = 10 #max 10 allowable components for time being - this doesn't apply any more....
        self.grid = False
        self.nodes = []
        self.comps = [] #list of component objects
        self.compsLoose = [] #list of detached components
        self.firstComponent = True #initially True
        self.componentCounter = 0 #start counting at 0

        self.realNodes = [] #physical nodes to pass to Jamesons program to measure voltage at
        self.equivalencies = [] #track which rows are the same node (connected by wires)
    
        self.wires = []
        self.powerNode = False
        self.groundNode = False

    def getGrid(self):
        return self.grid
    def getNodes(self):
        return self.nodes
    def getComps(self):
        return self.comps
    def getRealNodes(self): #THIS IS WHAT THE LIVE VOLTAGE MEASUREMENTS WILL TAKE AS INPUT
        return self.realNodes

    def setComps(self,compRow):
        (self.comps).append(compRow)
        #print self.comps
    def setNodes(self,nodeRow):
        (self.nodes).append(nodeRow)
        #print self.nodes
    def setPowerNode(self,powerNodeInput):
        self.powerNode = powerNodeInput
    def setGroundNode(self,groundNodeInput):
        self.groundInput = groundNodeInput


    def initialize(self):
        if (not((self.grid)and(self.nodes)and(self.comps))): #make sure this object has not already been initialized  
            self.grid = np.array([[0 for gridRows in range(self.n)] for gridCols in range(self.n)])
            
            #initNodeRow = [False,[0]*self.n,0,0,0] #T/F,n nodes, comp#, col#, row# <- can't use this, makes all items in list linked, can't change 1 without changing the others
            self.nodes = [[False,[0]*self.n,0,0,0] for nodesRows in range(self.c)]

            #initCompsRow = [False,0,0,0,0] #T/F, comp #, type&value OR reference,T/F - not anymore, just placing in component objects
            #####HERE - done
            self.comps = []
            print "Grid initialized with " + str(self.n+1) + "-by-" + str(self.n+1) + " matrix, max " + str(self.c) + " components." 

            #clean up
            gridRows = None
            gridCols = None
            initNodeRow = None
            nodes = None
            nodesRows = None
            initCompsRow = None
            compsRows = None

##            print self.comps
##            print self.nodes
            
            return

        else:
            print "This object has already been initialized!!!"
            return



    def addComponent(self,newComponent):
        if (self.firstComponent): #this is the first component to be added

            print "Adding first component!"
            self.firstComponent = False #there can be only one
            
            #get required information from component:
            compNode1 = newComponent.getNodeI()
            self.powerNode = compNode1
            compNode2 = newComponent.getNodeO()
            self.realNodes = [compNode1,compNode2]
            self.groundNode = compNode2
            compType = newComponent.getType()
            compValue = newComponent.getValue()

            #update object - MOVE TO BEFORE SELF.COMP CONFIG ONCE REDONE
            newComponent.setCompID(self.componentCounter)
            print "Component: " + str(self.componentCounter) + " added! (" + str(compType)+ " " + str(compValue) + ")"
            
            #add info to Comps
            #####HERE - done
            self.comps.append(newComponent) #first can be true so long as first component connected to power

            #add info to Nodes
            self.nodes[0][0] = True
            self.nodes[0][1][int(compNode1)] = 1
            self.nodes[0][1][int(compNode2)] = 1
            self.nodes[0][2] = self.componentCounter
            self.nodes[0][-2] = 0
            self.nodes[0][-1] = 0


            #add to grid
            self.grid[0][0] = self.componentCounter + 1

            self.componentCounter = self.componentCounter + 1

            #clean up
            compNode1 = None
            compNode2 = None
            compType = None
            compValue = None

            #print self.nodes
            #print self.comps
            #wanna run visualization thingy here
##            print self.nodes, self.comps
            print self.comps
            return

        else: #this is NOT the first component
            print "Adding additional component!"
            #get required information
            compNode1 = newComponent.getNodeI()
            
            compNode2 = newComponent.getNodeO()
        
            if not(compNode1 in self.realNodes):
                (self.realNodes).append(compNode1)
            if not(compNode2 in self.realNodes):
                (self.realNodes).append(compNode2)
            if ((not(compNode2 in self.realNodes)) and (not(compNode1 in self.realNodes))):
                #THIS COMPONENT IS NOT CONNECTED TO ANYTHING YET
                compIsLoose = True
                self.compsLoose.append(newComponent)
            else:
                compIsLoose = False
                newComponent.setCompConnected()
                                                
            compType = newComponent.getType()
            compValue = newComponent.getValue()

            compName = str(compType) + " " + str(compValue)

            if (compType == "wire"):
                #wires need to be handled differently
                (self.wires).append(newComponent)
                print "Wire added!"
                return

            #add info to Nodes
            
            self.nodes[self.componentCounter][0] = True
            self.nodes[self.componentCounter][1][compNode1] = 1
            self.nodes[self.componentCounter][1][compNode2] = 1
            self.nodes[self.componentCounter][2] = self.componentCounter

            #update object
            newComponent.setCompID(self.componentCounter)
            print "Component: " + str(self.componentCounter) + " added! (" + str(compType)+ " " + str(compValue) + ")"

            #CHECK nodes if component is RCD, wire (??) or gate - NOT full IC
            if ((newComponent.getType() == 'resistor') or (newComponent.getType() == 'capacitor') or (newComponent.getType() == 'diode') or (newComponent.getType() == 'wire') or (newComponent.getType() == 'gate')):
                otherList, diff = self.checkNodes(self.nodes[self.componentCounter],self.componentCounter)
                if (diff != 0):
                    self.groundNode = self.powerNode + diff

            #add info to Comps
            #HERE - shouldn't need this part anymore
            #self.comps[self.componentCounter][0] = True
            #self.comps[self.componentCounter][1] = self.componentCounter
            #self.comps[self.componentCounter][2] = compName
            self.comps.append(newComponent)
##            if looseComp: NEED TO FINISH THIS - NOT IS USE ANYMORE
##                self.comps[self.componentCounter][3] = False
##            else:
##                self.comps[self.componentCounter][3] = True

            #component finished
            self.componentCounter = self.componentCounter + 1
            
            #clean up
            compNode1 = None
            compNode2 = None
            compType = None
            compValue = None
            compName = None
            #end addComponent

            print self.comps
        
    def checkNodes(self,COMPARE_THIS_ROW, ROW_NUMBER):
        relationships = []
        
        for eachRow in self.nodes:
            #first col of each row will be True if there is a component stored there, don't compare same row
            #print eachRow[0]
            if (eachRow[0] and (eachRow[2] != ROW_NUMBER)):
                print "Comparing Nodes!"
                #then compare nodes list
                #print eachRow[1]
                #print COMPARE_THIS_ROW[1]
                multiplyCols = [cols1 * cols2 for (cols1, cols2) in zip(eachRow[1], COMPARE_THIS_ROW[1])]
                sumOfMultiplyCols = sum(multiplyCols)
                #print sumOfMultiplyCols
                if (sumOfMultiplyCols == 0):
                    #no action - no relation between rows
                    continue
                elif (sumOfMultiplyCols == 1):
                    print "In Series!"

                    #these two rows are in series
                    #check if True or False component
##                    if (self.comps[eachRow[2]][3]):
                    #in series after
##                    print eachRow
##                    print eachRow[-1]
##                    print eachRow[-2]
                    columnNew = eachRow[-1]
                    rowNew = eachRow[-2] + 1
                    self.nodes[self.componentCounter][-2] = rowNew
                    self.nodes[self.componentCounter][-1] = columnNew

                    countCols = -1
                    pointcheck = True
                    for eachCol in eachRow[1]:
                        countCols = countCols + 1
                        if ((eachCol + COMPARE_THIS_ROW[1][countCols]) == 1):
                            if pointcheck:
                                point1 = countCols
                                pointcheck = False
                            else:
                                point2 = countCols

                    #if (point2 > point1):
                    colsDiff = point2 - self.powerNode #!
                    break
                    #else:
                       # colsDiff = point1 - point2

                    #self.nodes[self.componentCounter,-1] = eachRow[-1] - 1
##                        relationships.append(["S",eachRow[2]])
##                    else:
##                        continue
                elif (sumOfMultiplyCols == 2):
                    print "In Parallel!"
                    
                    #####HERE
                    partInParallel = (self.comps[eachRow[2]]).getCompID() #
                    relationships.append(partInParallel)
                    #these two rows are in parallel
                    
##                  #check if True or False component
                    columnNew = eachRow[-1] + 1
                    rowNew = eachRow[-2]
                    self.nodes[self.componentCounter][-2] = rowNew #column position
                    self.nodes[self.componentCounter][-1] = columnNew #row position
                    colsDiff = 0
                    
        print "SELF.GRID"
        print self.grid


        if (ROW_NUMBER != 39):
            for counter in range(20):
                print "check"
                print counter
                print self.grid[columnNew+counter][rowNew]
                if (self.grid[columnNew+counter][rowNew] == 0):
                    self.grid[columnNew+counter][rowNew] = self.componentCounter + 1
                    break
                else:
                    continue

        print "SELF.GRID"
        print self.grid
        

        #if (self.comps[ROW_NUMBER].getCompConnected()):
            #check if component had no connections
            

        try:
            colsDiff
        except NameError:
            colsDiff = 0 #set some arbitrary value if unassigned by this point for secondary checking function
            
        return relationships, colsDiff
        #end checkNodes


    def checkDummy(self,inRail1,inRail2):
        print "CHECK DUMMY IS RUNNING"
        print self.nodes
        dummyRow = [True,[0]*self.n,0,0,0]
        dummyRow[1][int(inRail1)] = 1
        print "OUT OF BOUNDS??"
        print int(inRail2)
        dummyRow[1][int(inRail2)] = 1

        listThatMatters, other = self.checkNodes(dummyRow,39) #last number doens't matter so long as it's outside of possible component inventory
        print "LIST THAT MATTERS: "
        print listThatMatters
        listFormatted = []
        for eachItem in listThatMatters:
            splitString = eachItem.split()
            stringFormatted = [str(splitString[0]),float(splitString[1])]
            listFormatted.append(stringFormatted)
        print listFormatted
        return listFormatted

    def drawGrid(self):
        #####HERE
        PowerComp = Component()
        PowerComp.newComponent("Power","","source",0,0)
        self.comps.append(PowerComp)
        print PowerComp.getType()
        GroundComp = Component()
        GroundComp.newComponent("Ground","","source",0,0)
        self.comps.append(GroundComp)
        print GroundComp.getType()
        #self.comps[7] = [True,100,("Power"),True]
        #self.comps[8] = [True,200,("Ground"),True]
        
        currentGrid = self.grid
        
        try:
            mainpage
            for item in mainpage.items[:]:
                item.undraw()
        except:
            print "new mainpage"

        #remove all empty rows and cols
        currentGrid= np.delete(currentGrid,np.where(~currentGrid.any(axis=1))[0], axis=0)
        currentGrid= np.delete(currentGrid,np.where(~currentGrid.any(axis=0))[0], axis=1)

        print "CurrentGrid before drawing"
        print currentGrid
        #####HERE
        for eachComp in self.comps:
            typeAndValue = str(eachComp.getType()) + " " + str(eachComp.getValue())
            if ((typeAndValue == "diode forward") or (typeAndValue == "diode reverse")):
                compIDNum = eachComp.getCompID()+1
                print compIDNum
                print "compindex"
                placeholderList = currentGrid.tolist()
                print "placeholder list"
                print placeholderList
                rowNum = 0
                for eachItem in placeholderList:
                    try:
                        colNum = eachItem.index(compIDNum)
                        break
                    except:
                        rowNum = rowNum + 1
                if (colNum == 0):
                    #connected to power - check if node1 of diode = powerNode
                    if (eachComp.getNodeI() == self.powerNode):
                        print "no switch for diode"
                        if (eachComp.getValue() == "forward"):
                            eachComp.setValue("forward->")
                        else:
                            eachComp.setValue("<-reverse")
                        #no change required
                        continue
                    else:
                        print "diode switched"
                        #otherwise need to switch polarity
                        if (eachComp.getValue() == "forward->"):
                            eachComp.setValue("<-reverse")
                        else:
                            eachComp.setValue("forward->")
                            continue
                else:
                    #get component before
                    for scrollCount in range(20):
                        if (placeholderList[scrollCount][(colNum-1)] == 0):
                            continue
                        else:
                            attachedComp = placeholderList[scrollCount][(colNum-1)]
                            print "Component Previous: "
                            print attachedComp
                            break
                    if (self.comps[attachedComp-1].getNodeO() == eachComp.getNodeI()):
                        print "no switch for diode"
                        if (eachComp.getValue() == "forward"):
                            eachComp.setValue("forward->")
                        else:
                            eachComp.setValue("<-reverse")
                        #no change required
                        continue
                    else:
                        print "diode switched"
                        #otherwise need to switch polarity
                        if (eachComp.getValue() == "forward->"):
                            eachComp.setValue("<-reverse")
                        else:
                            eachComp.setValue("forward->")
                            continue
                    ##self.comps[attachedComp][
                    

        dimX = 500.0
        dimY = 500.0

        rows,cols = currentGrid.shape

        print "NODES"
        print self.powerNode
        print self.groundNode

        if (self.wires != []):
            if (((self.wires[0].getNodeI() == self.powerNode) or (self.wires[0].getNodeI() == self.groundNode)) and ((self.wires[0].getNodeO() == self.powerNode) or (self.wires[0].getNodeO() == self.groundNode))):
                print "WARNING - WIRE CONNECTING POWER AND GROUND"

        print "FLAG 1"
        print currentGrid
        
        middleRow = rows/2
        currentGrid = np.insert(currentGrid, 0, 0, axis=1)
        currentGrid[0][0] = self.componentCounter + 1
        currentGrid = np.insert(currentGrid, cols+1, 0, axis=1)
        currentGrid[0][cols+1] = self.componentCounter + 2

        print "FLAG 2"
        print currentGrid

        if (rows == 0):
            print "only 1 row"
        else:
            rowMultiplier = dimX/(rows)/2.0
            colMultiplier = dimY/(cols+2)/2.0
            
        mainpage = GraphWin("CircuitDiagram",dimX,dimY)

        V = []
        nodeCounter = 0
        
        for i in range(0,10): #randomly generated voltages
            x = random.randint(1,10)
            V.append(x)

        current = random.randint(0,50) #randomly generated current
    
        rowCount = -1
        currentGridRow = -1
        for eachRow in currentGrid:
            
            colCount = 1
            rowCount = rowCount + 2
            currentGridRow = currentGridRow + 1
            currentGridCol = 0
            
            for eachCol in eachRow:

                print "At position: " + str(rowCount) + "," + str(colCount)
                if (currentGrid[currentGridRow][currentGridCol] == 0):
                    colCount = colCount + 2
                    currentGridCol = currentGridCol + 1
                    continue


                if ((rowCount > 1) and (colCount>1) and (colCount<10)): #IF LINES ARE NOT SHOWING UP, TRY MODIFYING THESE
                    #draw parallel lines between current and previous component
                    lineRowPos = (rowCount*rowMultiplier)
                    newLineRowPos = (rowCount-2)*rowMultiplier
                    lineColPos = (colCount*colMultiplier - colMultiplier)
                    newLineColPos = (colCount*colMultiplier + colMultiplier)
        
                    Point1 = Point(lineColPos,newLineRowPos)
                    Point2 = Point(lineColPos,lineRowPos)
                    drawLine = Line(Point1,Point2)
                    drawLine.draw(mainpage)
                
                    Point3 = Point(newLineColPos,lineRowPos)
                    Point4 = Point(newLineColPos,newLineRowPos)
                    drawLine = Line(Point3,Point4)
                    drawLine.draw(mainpage)
                    
                #if (colCount > 1):
                    #draw line between current and last component
                    #not neccessary at moment....will be in future
                #HERE
                print "WHAT'S THIS"
                print self.comps[eachCol-1].getCompID()
                print eachCol
                compName = (str(self.comps[eachCol-1].getType())[0]) + " " + str((self.comps[eachCol-1].getValue()))
                print compName
                if (compName == 0): #if there's nothing there, skip - NOT SURE THIS IS DOING ANYTHING ANYMORE, BUT WHATEVS
                    print "SKIP"
                    colCount = colCount + 2
                    currentGridCol = currentGridCol + 1
                    continue

                drawComp = Rectangle(Point(colCount*colMultiplier-20,rowCount*rowMultiplier+10), Point(colCount*colMultiplier+20,rowCount*rowMultiplier-10))
                drawComp.draw(mainpage)
                labelComp = Text(Point(colCount*colMultiplier,rowCount*rowMultiplier+20),compName)
                labelComp.draw(mainpage)

                Point1 = Point(colCount*colMultiplier-colMultiplier,rowCount*rowMultiplier)
                Point2 = Point(colCount*colMultiplier-30,rowCount*rowMultiplier)
                            
                drawLine = Line(Point1,Point2)
                drawLine.draw(mainpage)
                
                Point3 = Point(colCount*colMultiplier+30,rowCount*rowMultiplier)
                Point4 = Point(colCount*colMultiplier+colMultiplier,rowCount*rowMultiplier)
                drawLine = Line(Point3,Point4)
                drawLine.draw(mainpage)
                
                colCount = colCount + 2
                currentGridCol = currentGridCol + 1

        for eachCol in range(1,colCount/2):
            if (eachCol == 1):
                currentPoint = Point(eachCol*colMultiplier*2-colMultiplier,480)
                textCurrent = Text(currentPoint,"Current: " + str(current))
                ##textCurrent.draw(mainpage) #inactive - for now we're NOT utilizing current measurements
            textPoint = Point(eachCol*colMultiplier*2,20)
            drawNode =  Text(textPoint,V[nodeCounter]) 
            nodeCounter = nodeCounter + 1
            ##drawNode.draw(mainpage) #inactive - for now we're reading voltage readings from command line

        click = mainpage.getMouse()
        mainpage.close()

        #####HERE
        self.comps.remove(PowerComp) #remove power and ground components from comp list
        self.comps.remove(GroundComp) 
                
        #end drawgrid
        
        
#FOR TESTING!!
testCase = GridAndNodesAndComps()
testCase.initialize()

##addedComponent = Component()
##addedComponent.newComponent("IC","7408","IC",0,13)
##testCase.addComponent(addedComponent)
##
##print addedComponent.getSuperComp()
#need to initialize ICs as well?
##if (addedComponent.getSuperComp()):
##    print "Component is super component - " + str(addedComponent.getCompID())
##    #if supercomp - need to initialize other components as well
##    totalPins = int(addedComponent.getNodeO()) - int(addedComponent.getNodeI())
##    totalGates = (addedComponent.getMakeUp())
##    print "Made up of: " + str(totalGates) + " gates"
##    jumpValue = (totalPins)/int(totalGates)
##    logicInputType = addedComponent.getNodeConfig()
##    if (logicInputType == "2in1out"):
##        for eachSingleGate in range(1,totalPins-2,jumpValue):        
##            gateComponent = Component()
##            gateComponent.newComponent("gate",addedComponent.getValue(),addedComponent.getNodeConfig(),eachSingleGate,0)
##            testCase.addComponent(gateComponent)
##            gateComponent.setParentComp(addedComponent.getCompID())
##            addedComponent.addSubComponent(gateComponent)
##            testCase.getNodes()
##    elif (logicInputType == "2in-1out"):
##        for eachSingleGate in range(2,totalPins-2,jumpValue):
##            gateComponent = Component()
##            gateComponent.newComponent("gate",addedComponent.getValue(),addedComponent.getNodeConfig(),eachSingleGate,0)
##            testCase.addComponent(gateComponent)
##            gateComponent.setParentComp(addedComponent.getCompID())
##            addedComponent.addSubComponent(gateComponent)
            


#print testCase.getNodes()

##
##testCase.drawGrid()
####
##secondComponent = Component()
##secondComponent.newComponent("resistor","100","default",4,3)
##testCase.addComponent(secondComponent)
##print testCase.getRealNodes()
##
##testCase.drawGrid()
##

##
##
##testCase.drawGrid()
####

##
##firstComponent = Component()
##firstComponent.newComponent("resistor","2000","default",1,3)
##testCase.addComponent(firstComponent)
##
##testCase.drawGrid()
##
##secondComponent = Component()
##secondComponent.newComponent("capacitor","10","default",3,6)
##testCase.addComponent(secondComponent)
##
##testCase.drawGrid()
##
##thirdComponent = Component()
##thirdComponent.newComponent("diode","forward","default",3,6)
##testCase.addComponent(thirdComponent)
##
##testCase.drawGrid()

##fourthComponent = Component()
##fourthComponent.newComponent("diode","forward","default",6,10)
##testCase.addComponent(fourthComponent)
##
##testCase.drawGrid()
##
##fifthComponent = Component()
##fifthComponent.newComponent("capacitor","10","default",6,10)
##testCase.addComponent(fifthComponent)
##
##testCase.drawGrid()

##fifthComponent = Component()
##fifthComponent.newComponent("resistor","9000","default",3,10)
##testCase.addComponent(fifthComponent)
##
##testRailCheck = testCase.checkDummy(3,10)



##


#need to add dedicated checking module - check for any circuit with no resistance (wire, C or D alone)
#need to redo self.comps - just store comp objects!!! - will make diode implementation easier
#do more clean up
