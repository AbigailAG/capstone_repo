
import numpy as np

class Component:

    def __init__(self):

        self.type = "default"
        self.value = False
        self.nodeconfig = False
        self.nodeI = False
        self.nodeO = False
        self.node1 = False
        self.node2 = False
        self.node3 = False
        self.node4 = False
        
        
    def getType(self):
        return self.type
    def getValue(self):
        return self.value
    def getNodeI(self):
        return self.nodeI
    def getNodeO(self):
        return self.nodeO
    def getNodeconfig(self):
        return self.noteconfig
    def getNode1(self):
        return self.node1
    def getNode2(self):
        return self.node2
    def getNode3(self):
        return self.node3
    def getNode4(self):
        return self.node4
    

    def newComponent(self,compType,compValue,compNodeConfig,compNodeI, compNodeO,compNodesOther):
        if (self.type == "default"): #if the component has not been initialised
            #add checks for proper type and values HERE
            
            #types = power (1), ground (1), resistor (n), etc....
            checkResult = self.checkType(compType)
            if checkResult:
                self.type = compType
            else:
                print "Process failed...."
                return
            self.value = compValue
            self.nodeconfig = compNodeConfig
            self.nodeI = compNodeI
            self.nodeO = compNodeO
            self.node1 = compNodesOther[0]
            self.node2 = compNodesOther[1]
            self.node3 = compNodesOther[2]
            self.node4 = compNodesOther[3]
        else:
            print "this component has already been initialized..."
            return


    def checkType(self,inputType):
        print inputType
        
        RCDList = ["R", "D","C","W"]
        
        if (inputType in RCDList):
            result = True
        else:
            print "Invalid component type! - fail"
            result = False


        return result
            
            


R500 = Component()
R500.newComponent("R",500,"default",1,4,[0,0,0,0])
R700 = Component()
R700.newComponent("R",700,"default",4,6,[0,0,0,0])
R1100 = Component()
R1100.newComponent("R",1100,"default",1,6,[0,0,0,0])

#Power = Component()
#Power.newComponent("P",5,"power",1,False,[0,0,0,0])






        
            
            
        






        
