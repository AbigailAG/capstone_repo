import numpy as np

class Component:

    def __init__(self):

        self.type = "default" #initially default - set as comp type (either RCD, wire, or IC)
        self.value = False #value of comp - set as RCD value, n/a for wire, part descriptor for IC
        self.nodeConfig = False #whether any special instructions are required - for ICs
        self.superComp = False #is this component made up of other components? I.E. series elements, total ICs (true or false)
        self.nodeI = False #"input" for RCD/wire -> and power for total IC 
        self.nodeO = False #"output" for RCD/wire -> and ground for total IC
        self.compConnected = False #is the comp part of the circuit - true? or loose - false?
        self.compID = False #given ID when added to grid

        self.node1 = False #(first) input for single IC (all will have at least one)
        self.node2 = False #second input for single IC (if neccesary)
        self.node3 = False #output for single IC
        self.makeUp = 0 #if super component, number of component it's made up of
        self.subComponents = [] #list of component it's made up of
        self.parentComp = False #supercomp it belongs to if it is part of a supercomponent

        self.inputType = False #if super component IC - how many inputs?
        self.outputType = False #where is the output relative to the inputs? (i.e. in some cases it's after the two, sometimes before) - this should be 1 or -1
        self.compActive = False #if the comp requires additional power/ground, show if those have been provided

        #diode specific
        self.compPolarity = False
        
    def getType(self):
        return self.type
    def getValue(self):
        return self.value
    def setValue(self, newValue):
        self.value = newValue
    def getNodeConfig(self):
        return self.nodeConfig
    def getSuperComp(self):
        return self.superComp
    def getNodeI(self):
        return self.nodeI
    def getNodeO(self):
        return self.nodeO
    def getCompConnected(self):
        return self.compConnected
    def setCompConnected(self):
        self.compConnected = True
    def getCompID(self):
        return self.compID
    def setCompID(self, IDnum):
        self.compID = IDnum        
    
#IC specials
    def getNode1(self):
        return self.node1
    def getNode2(self):
        return self.node2
    def getNode3(self):
        return self.node3
    def getMakeUp(self):
        #add check to see if supercomponent before making this call
        return self.makeUp
    def getSubComponents(self):
        return self.subComponents
    def addSubComponent(self,subComp):
        subCompID = subComp.getCompID()
        (self.subComponents).append(subCompID)
        print "Sub Component added to SuperComp!"
    def getParentComp(self):
        return self.parentComp
    def setParentComp(self,parentIDNum):
        self.parentComp = parentIDNum
        print "Parent component set! (parent: " + str(parentIDNum)+ " daughter: " + str(self.compID)
    def getInputType(self):
        return self.inputType
    def getOutputType(self):
        return self.outputType
    def getCompActive(self):
        return self.compActive

#diode specials - NOT USING THESE ANYMORE
    def getCompPolarity(self):
        return self.compPolarity
    def setCompPolarity(self,inputPolarity):
        self.compPolarity = inputPolarity
        print "Polarity set as: " + str(inputPolarity)



    def newComponent(self,compType,compValue,compNodeConfig,compNodeI, compNodeO):
        if (self.type == "default"): #if the component has not been initialised
            #add checks for proper type and values HERE
            if (compNodeConfig == "default"):
                  #types = power (1), ground (1), resistor (n), etc....
                checkResult = self.checkType(compType)
                if checkResult:
                    self.type = compType
                else:
                    print "Process failed...."
                    return
                self.value = compValue
                self.nodeConfig = compNodeConfig
                self.nodeI = compNodeI
                self.nodeO = compNodeO
            elif (compNodeConfig == "source"):
                    if (compType == "Power"):
                        self.type = "Power"
                        self.compID = 100
                        self.value = ""
                    elif (compType == "Ground"):
                        self.type = "Ground"
                        self.compID = 200
                        self.value = ""

            elif (compNodeConfig == "IC"):
                #it's an total IC, determine specs
                textfilename = "IClist.txt"
                infile = open(textfilename,'r')
                infile.close
                IClist = infile.readlines()
                for eachIC in IClist:
                    eachICArray = eachIC.split()
                    print eachICArray[0]
                    if (eachICArray[0] == compValue):
                        print "IC on file!"
                        #total IC setup
                        self.type = eachICArray[0] #ID i.e. 7408
                        self.value = eachICArray[1] #type i.e. AND
                        self.nodeConfig = eachICArray[5] #input/output config - 2in1out,2in-1out
                        self.superComp = True
                        self.nodeI = eachICArray[3]
                        self.nodeO = eachICArray[4]
                        self.makeUp = eachICArray[2] #how many single components
                        self.compActive = False
                        #create individual components
            #it's a single IC                
            elif (self.nodeConfig == "2in1out"):
                self.node1 = compNodeI #from IC
                self.node2 = compNodeI + 1 #from IC
                self.node3 = int(self.node2) + 1 #output is next to second input (I I O)
                self.type = "gate"
                self.value = compValue #from IC
                self.nodeCofig = compNodeConfig #from IC
                self.superComp = False
            elif (self.nodeConfig == "2in-1out"):
                self.node1 = compNodeI
                self.node2 = compNodeI + 1
                self.node3 = int(self.node1) - 1 #output is next to first input (O I I)
                self.type = "gate"
                self.value = compValue #from IC
                self.nodeCofig = compNodeConfig #from IC
                self.superComp = False
            elif (self.nodeConfig == "1in1out"):
                self.nodeI = compNodeI #one input and one input can use nodeI and nodeO as usual
                self.nodeO = compNodeI + 1
                self.type = "gate"
                self.value = compValue #from IC
                self.nodeCofig = compNodeConfig #from IC
                self.superComp = False
            elif (self.nodeConfig == "1in-1out"):
                self.nodeI = compNodeI
                self.nodeO = compNodeI - 1
                self.type = "gate"
                self.value = compValue #from IC
                self.nodeCofig = compNodeConfig #from IC
                self.superComp = False
            
                
        else:
            print "this component has already been initialized..."
            return


    def checkType(self,inputType):
        print inputType

        #components allowed at the moment are: resistors, diodes, capacitors, wires, sources (power and ground) // ICs are handled separately
        allowedList = ["resistor", "diode","capacitor","wire","source"]
        
        if (inputType in allowedList):
            result = True
        else:
            print "Invalid component type! - fail"
            result = False


        return result
            
            


##R500 = Component()
##R500.newComponent("R",500,"default",1,4,[0,0,0,0])
##R700 = Component()
##R700.newComponent("R",700,"default",4,6,[0,0,0,0])
##R1100 = Component()
##R1100.newComponent("R",1100,"default",1,6,[0,0,0,0])

#Power = Component()
#Power.newComponent("P",5,"power",1,False,[0,0,0,0])






        
            
            
        




