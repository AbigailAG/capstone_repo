
import random
import numpy as np
from Component import *
from graphics import *

class GridAndNodesAndComps:

    def __init__(self):
        self.n = 15 #total of 7 rows and columns in grid - MUST BE ODD   
        self.c = 10 #max 10 allowable components for time being
        self.grid = False
        self.nodes = []
        self.comps = []
        self.firstComponent = True
        self.componentCounter = 0 #start counting at 0

        self.wires = []
        self.powerNode = False
        self.groundNode = False

    def getGrid(self):
        return self.grid
    def getNodes(self):
        return self.nodes
    def getComps(self):
        return self.comps

    def setComps(self,compRow):
        (self.comps).append(compRow)
        #print self.comps
    def setNodes(self,nodeRow):
        (self.nodes).append(nodeRow)
        #print self.nodes


    def initialize(self):
        if (not((self.grid)and(self.nodes)and(self.comps))): #make sure this object has not already been initialized  
            self.grid = np.array([[0 for gridRows in range(self.n)] for gridCols in range(self.n)])
            
            #initNodeRow = [False,[0]*self.n,0,0,0] #T/F,n nodes, comp#, col#, row# <- can't use this, makes all items in list linked, can't change 1 without changing the others
            self.nodes = [[False,[0]*self.n,0,0,0] for nodesRows in range(self.c)]

            #initCompsRow = [False,0,0,0,0] #T/F, comp #, type&value OR reference,T/F  
            self.comps = [[False,0,0,0] for compsRows in range(20)]
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
            self.firstComponent = False #there can only be one
            
            #get required information from component:
            compNode1 = newComponent.getNodeI()
            self.powerNode = compNode1
            compNode2 = newComponent.getNodeO()
            self.groundNode = compNode2
            compType = newComponent.getType()
            compValue = newComponent.getValue()

            #add info to Comps
            self.comps[0] = [True,self.componentCounter,(str(compType)+str(compValue)),True]

            #add info to Nodes
            self.nodes[0][0] = True
            self.nodes[0][1][compNode1] = 1
            self.nodes[0][1][compNode2] = 1
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
            return

        else: #this is NOT the first component

            print "Adding additional component!"
            #get required information
            compNode1 = newComponent.getNodeI()
            
            compNode2 = newComponent.getNodeO()
            compType = newComponent.getType()
            compValue = newComponent.getValue()

            compName = str(compType) + str(compValue)

            if (compType == "W"):
                #wires need to be handled differently
                (self.wires).append(newComponent)
                print "Wire added!"
                return 


            #add info to Nodes
            self.nodes[self.componentCounter][0] = True
            self.nodes[self.componentCounter][1][compNode1] = 1
            self.nodes[self.componentCounter][1][compNode2] = 1
            self.nodes[self.componentCounter][2] = self.componentCounter
            
            #CHECK nodes
            diff = self.checkNodes(self.nodes[self.componentCounter],self.componentCounter)
            if (diff != 0):
                self.groundNode = self.powerNode + diff 

            #add info to Comps
            self.comps[self.componentCounter][0] = True
            self.comps[self.componentCounter][1] = self.componentCounter
            self.comps[self.componentCounter][2] = compName
            self.comps[self.componentCounter][3] = True

            #component finished
            self.componentCounter = self.componentCounter + 1
            
            #clean up
            compNode1 = None
            compNode2 = None
            compType = None
            compValue = None
            compName = None

        



    def checkNodes(self,COMPARE_THIS_ROW, ROW_NUMBER):

##        relationships = []
        
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
                    colsDiff = point2 - self.powerNode
                    #else:
                       # colsDiff = point1 - point2

                            
                            
                    
                    #self.nodes[self.componentCounter,-1] = eachRow[-1] - 1
##                        relationships.append(["S",eachRow[2]])
##                    else:
##                        continue
                elif (sumOfMultiplyCols == 2):
                    print "In Parallel!"
                    #these two rows are in parallel
                    
##                    #check if True or False component
                    columnNew = eachRow[-1] + 1
                    rowNew = eachRow[-2]
                    self.nodes[self.componentCounter][-2] = rowNew #column position
                    self.nodes[self.componentCounter][-1] = columnNew #row position
                    colsDiff = 0


                if (self.grid[columnNew][rowNew] == 0):
                    self.grid[columnNew][rowNew] = self.componentCounter + 1

        return colsDiff






    def drawGrid(self):


        self.comps[7] = [True,100,("Power"),True]
        self.comps[8] = [True,200,("Ground"),True]
        
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

        dimX = 500.0
        dimY = 500.0

        rows,cols = currentGrid.shape

        print "NODES"
        print self.powerNode
        print self.groundNode


        if (self.wires != []):
            if (((self.wires[0].getNodeI() == self.powerNode) or (self.wires[0].getNodeI() == self.groundNode)) and ((self.wires[0].getNodeO() == self.powerNode) or (self.wires[0].getNodeO() == self.groundNode))):
                print "WARNING - WIRE CONNECTING POWER AND GROUND"
        

        middleRow = rows/2
        currentGrid = np.insert(currentGrid, 0, 0, axis=1)
        currentGrid[middleRow][0] = 8
        currentGrid = np.insert(currentGrid, cols+1, 0, axis=1)
        currentGrid[middleRow][cols+1] = 9

        if (rows == 0):
            print "only 1 row"
        else:
            rowMultiplier = dimX/(rows)/2.0
            colMultiplier = dimY/(cols+2)/2.0
            
        mainpage = GraphWin("CircuitDiagram",dimX,dimY)


        V = []
        nodeCounter = 0
        

        for i in range(0,10):
            x = random.randint(1,10)
            V.append(x)

        current = random.randint(0,50)
    

        rowCount = -1
        for eachRow in currentGrid:
            
            colCount = 1
            rowCount = rowCount + 2
            
            for eachCol in eachRow:

                if ((rowCount > 1) and (colCount>1) and (colCount<4)):
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
                compName = self.comps[eachCol-1][2]
                if (compName == 0):
                    colCount = colCount + 2
                    continue

                drawComp = Text(Point(colCount*colMultiplier,rowCount*rowMultiplier),compName)
                drawComp.draw(mainpage)

                Point1 = Point(colCount*colMultiplier-colMultiplier,rowCount*rowMultiplier)
                Point2 = Point(colCount*colMultiplier-30,rowCount*rowMultiplier)
                
                
                drawLine = Line(Point1,Point2)
                drawLine.draw(mainpage)
                
                Point3 = Point(colCount*colMultiplier+30,rowCount*rowMultiplier)
                Point4 = Point(colCount*colMultiplier+colMultiplier,rowCount*rowMultiplier)
                drawLine = Line(Point3,Point4)
                drawLine.draw(mainpage)
                
                colCount = colCount + 2

        for eachCol in range(1,colCount/2):
            if (eachCol == 1):
                currentPoint = Point(eachCol*colMultiplier*2-colMultiplier,480)
                textCurrent = Text(currentPoint,"Current: " + str(current))
                textCurrent.draw(mainpage)
            textPoint = Point(eachCol*colMultiplier*2,20)
            drawNode =  Text(textPoint,V[nodeCounter])
            nodeCounter = nodeCounter + 1
            drawNode.draw(mainpage)

        click = mainpage.getMouse()
        mainpage.close()

        self.comps[-1] = [False,0,0,0]
        self.comps[-2] = [False,0,0,0]
                
        
        
        
###FOR TESTING!!
##
##testCase = GridAndNodesAndComps()
##testCase.initialize()
##
##firstComponent = Component()    
##firstComponent.newComponent("R","9.95k","default",2,4,[0,0,0,0])
##
##testCase.addComponent(firstComponent)
##
##testCase.drawGrid()
##
##secondComponent = Component()
##
##secondComponent.newComponent("R","9.89k","default",4,3,[0,0,0,0])
##
##testCase.addComponent(secondComponent)
##
##testCase.drawGrid()
##
##thirdComponent = Component()
##thirdComponent.newComponent("R","6.8k","default",3,10,[0,0,0,0])
##
##testCase.addComponent(thirdComponent)
##
##
##testCase.drawGrid()
##
##fourthComponent = Component()
##fourthComponent.newComponent("W","no_value","default",2,10,[0,0,0,0])
##
##testCase.addComponent(fourthComponent)
##
##testCase.drawGrid()




