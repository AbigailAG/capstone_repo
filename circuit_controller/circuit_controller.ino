/* Arduino Mega Pin Layout: (Note: A0-A15 = 54-69)
 * 0-1:     Disabled for serial communications
 * 2:       Run circuit power enable (1) / disable (0)
 *            If 0: Connect characterization circuit
 *                  Disconnect breadboard power rails
 *            If 1: Disconnect characterization circuit
 *                  Connect breadboard power rails
 * 3:       Demux enable. 
 * 4-9:     Demux select. LSB at 4
 * 10:      Mux enable. 
 * 11-16:   Mux select. LSB at 11
 * 17:      Reference resistor enable
 * 22-24:   Reference resistor select. LSB at 18
 * 20-21:   Disabled for SDA-SCL communication
 * 54(A0):  Rail voltage analog read
 * 55-71(A1-A7):  Reference resistor analog read
 * TODO CURRENT PROBE
 * TODO DIGITAL POT FOR VOLTAGE
 */
#include <SPI.h>

#define PIN_CIRCUIT_POWER 2
#define PIN_DEMUX_ENABLE 3
#define PIN_DEMUX_SELECT_START 4
#define PIN_DEMUX_SELECT_LENGTH 6
#define PIN_MUX_ENABLE 10
#define PIN_MUX_SELECT_START 11
#define PIN_MUX_SELECT_LENGTH 6
#define PIN_REFERENCE_RESISTOR_ENABLE 17
#define PIN_COMP_INTERRUPT 18
#define PIN_REFERENCE_RESISTOR_SELECT_START 22
#define PIN_REFERENCE_RESISTOR_SELECT_LENGTH 3
#define PIN_LIN_POT_CS_BAR 25
#define PIN_LOG_POT_ENABLE 26
#define PIN_SPI_SCK 27
#define PIN_SPI_SDI 28
#define PIN_LIN_POT_RR_CS_BAR 29
#define PIN_COMP_READ 30
#define PIN_RAIL_ENABLE 31
#define PIN_RAIL_SELECT_START 32
#define PIN_RAIL_SELECT_LENGTH 3
#define PIN_RAIL_ANALOG_READ A0
#define PIN_REFERENCE_RESISTOR_ANALOG_READ_START A1
#define PIN_REFERENCE_RESISTOR_ANALOG_READ_LENGTH 4
//#define PIN_CURRENT_ANALOG_READ_HIGH A4
//#define PIN_CURRENT_ANALOG_READ_LOW A5
#define PIN_RAIL_READ_START A6
#define PIN_RAIL_READ_LENGTH 4

//#define PIN_LIN_POT_SCK 26 
//#define PIN_LIN_POT_SDI 27
//#define PIN_LIN_POT_SDO 28
//#define PIN_LIN_POT_SHDN_BAR 29
//#define PIN_LIN_POT_WP_BAR 30
//#define PIN_LIN_POT_1_A
//#define PIN_LIN_POT_1_B
//#define PIN_LIN_POT_1_W
//#define PIN_LIN_POT_1_A
//#define PIN_LIN_POT_1_B
//#define PIN_LIN_POT_1_W

#define PINS_DIGITAL_OUT {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25,26,27,28,29,31,32,33,34,51,52,53}

// Miscellanous constants
#define UNKNOWN_PIN 0xFF
#define BAUD_RATE 2000000
#define ADC_SIZE 1024
#define NUM_PINS 70
#define NUM_DIGITAL_PINS 54
#define NUM_ANALOG_PINS 16

// Arduino mode constants and state variable
int mode; 
#define MODE_INIT 0
#define MODE_BUILD 1
#define MODE_RUN 2
#define MODE_DEBUG 255

// Constants and variables relating to serial communication
#define READ_BUFFER_SIZE 10
int readBuffer[READ_BUFFER_SIZE]; 
int readBufferIndex; 
#define SERIAL_VALID_REQUEST    1
#define SERIAL_INVALID_REQUEST  0

// All available commands: 
// In any mode: 
#define SERIAL_CHANGE_MODE  255
// Init mode: 
#define SERIAL_INIT_CHARACTERIZE  2
#define SERIAL_INIT_ACKNOWLEDGE   1
#define SERIAL_INIT_PARALLEL_RESISTANCE 3
#define SERIAL_INIT_PARALLEL_CAPACITANCE 4
// Build mode: 
#define SERIAL_BUILD_CHANGE_REFERENCE_RESISTOR  19
#define SERIAL_BUILD_VOLTAGE_READ               2
#define SERIAL_BUILD_VOLTAGE_READ_DELAY         3
#define SERIAL_BUILD_VOLTAGE_TIME_READ          4
#define SERIAL_BUILD_TIME_READ                  5
#define SERIAL_BUILD_DIGITAL_COMP               6
#define SERIAL_BUILD_DIGITAL_INTERRUPT          7
// Run mode: 
#define SERIAL_RUN_POWER    1
#define SERIAL_RUN_VOLTAGE  2
#define SERIAL_RUN_CURRENT  3
// Debug mode: 
#define SERIAL_DEBUG_POWER                      1
#define SERIAL_DEBUG_DEMUX_ENABLE               2
#define SERIAL_DEBUG_DEMUX_SELECT               3
#define SERIAL_DEBUG_MUX_ENABLE                 4
#define SERIAL_DEBUG_MUX_SELECT                 5
#define SERIAL_DEBUG_REFERENCE_RESISTOR_ENABLE  6
#define SERIAL_DEBUG_REFERENCE_RESISTOR_SELECT  7
#define SERIAL_DEBUG_ENABLE_SHUTDOWN            8
#define SERIAL_DEBUG_READ_RAIL                  9
#define SERIAL_DEBUG_READ_CURRENT               10
#define SERIAL_DEBUG_READ_REFERENCE_RESISTOR    11
#define SERIAL_DEBUG_SET_PIN                    12
#define SERIAL_DEBUG_GET_PIN_DIRECTION          13
#define SERIAL_DEBUG_GET_PIN_STATE              14
#define SERIAL_DEBUG_LIN_POT_WRITE              15
#define SERIAL_DEBUG_LIN_POT_READ               16
#define SERIAL_DEBUG_LIN_POT_INCREMENT          17
#define SERIAL_DEBUG_LIN_POT_DECREMENT          18
#define SERIAL_DEBUG_LIN_POT_WRITE_2            19
#define SERIAL_DEBUG_LOG_POT_WRITE              20
#define SERIAL_DEBUG_LIN_POT_RR_WRITE           21
#define SERIAL_DEBUG_LIN_POT_RR_READ            22
#define SERIAL_DEBUG_LIN_POT_RR_INCREMENT       23
#define SERIAL_DEBUG_LIN_POT_RR_DECREMENT       24
#define SERIAL_DEBUG_RAIL_ENABLE                25
#define SERIAL_DEBUG_RAIL_SELECT                26
#define SERIAL_DEBUG_MUX_READ_RAIL              27

// Constants relating to mux sizes and locations
#define MUX_SELECT_SIZE 64
#define REFERENCE_RESISTOR_SELECT_SIZE 4
#define REFERENCE_RESISTOR_POT_INDEX 6
#define REFERENCE_RESISTOR_DRAIN_INDEX 7

// Constants related to pin enable settings
#define SWITCH_OFF      0
#define SWITCH_ON       1
#define SWITCH_TOGGLE   255

// Constants to standardize interpretations of forward and reverse bias for diodes
#define FORWARD_BIAS  1
#define REVERSE_BIAS  -1

// Constants related to initial characterization of board
#define INIT_NUM_MEASUREMENTS 10 
#define INIT_SWITCH_DELAY_US 1000
#define INIT_READ_DELAY_US 1000

// Variables to hold which rails are currently selected
int demuxRailSelected = 0; 
int muxRailSelected = 0; 
int referenceResistorRailSelected = 0; 
int readRailSelected = 0; 

#define LIN_POT_WRITE_COMMAND 00
#define LIN_POT_READ_COMMAND 3
#define LIN_POT_INCREMENT_COMMAND 01
#define LIN_POT_DECREMENT_COMMAND 2

unsigned long interruptTime = 0; 
int interruptTrigger = 0; 

// 2 methods of reading voltage during Run mode: 
//   1. (true) Always keep at highest ADC read. Low accuracy but not likely to blow Arduino
//   2. (false) Adjust reading voltage when reading to maximize precision at low voltages
bool robustRead = false; 



void setup() {
  //for (int i = 2; i <= 24; i++) {
  //  pinMode(i, OUTPUT); 
  //  digitalWrite(i, LOW); 
  //}
  int pinsDigitalOut[] = PINS_DIGITAL_OUT; 
  for (int i = 0; i < sizeof(pinsDigitalOut)/sizeof(int); i++) {
    pinMode(pinsDigitalOut[i], OUTPUT); 
    digitalWrite(pinsDigitalOut[i], LOW); 
  }
  digitalWrite(PIN_LIN_POT_CS_BAR, HIGH); 
  digitalWrite(PIN_LOG_POT_ENABLE, LOW); 
  for (int i = 54; i <= 57; i++) {
    pinMode(i, INPUT); 
  }
  pinMode(PIN_COMP_READ, INPUT_PULLUP); 
  pinMode(PIN_COMP_INTERRUPT, INPUT_PULLUP); 
  Serial.begin(BAUD_RATE); 
  mode = MODE_INIT; 
  readBufferIndex = 0; 
  // TODO
  //analogReference(INTERNAL2V56); 
  // NOTE: DO NOT EVER SWITCH THE ANALOG REFERENCE TO INTERNAL, OR YOU WILL BLOW THE ARDUINO
  analogReference(EXTERNAL); 
  //analogReference(DEFAULT);
  SPI.begin(); 

  
  unsigned int bitstream = 0; 
  bitstream = (bitstream<<4) + 0; 
  bitstream = (bitstream<<2) + LIN_POT_WRITE_COMMAND; 
  bitstream = bitstream<<2; 
  //bitstream = (bitstream<<2) + 1; //Enable wiper
  bitstream = (bitstream<<8) + 256; 
  //bitstream = (bitstream<<10) + data; 
  digitalWrite(PIN_LIN_POT_CS_BAR, LOW); 
  //delay(1000); 
  SPI.transfer16(bitstream); 
  digitalWrite(PIN_LIN_POT_CS_BAR, HIGH); 
  bitstream = 0; 
  bitstream = (bitstream<<4) + 1; 
  bitstream = (bitstream<<2) + LIN_POT_WRITE_COMMAND; 
  bitstream = bitstream<<2; 
  //bitstream = (bitstream<<2) + 1; //Enable wiper
  bitstream = (bitstream<<8) + 256; 
  //bitstream = (bitstream<<10) + data; 
  digitalWrite(PIN_LIN_POT_CS_BAR, LOW); 
  //delay(1000); 
  SPI.transfer16(bitstream); 
  digitalWrite(PIN_LIN_POT_CS_BAR, HIGH);
  bitstream = 0; 
  bitstream = (bitstream<<4) + 0; 
  bitstream = (bitstream<<2) + LIN_POT_WRITE_COMMAND; 
  bitstream = bitstream<<2; 
  //bitstream = (bitstream<<2) + 1; //Enable wiper
  bitstream = (bitstream<<8) + 256; 
  //bitstream = (bitstream<<10) + data; 
  digitalWrite(PIN_LIN_POT_RR_CS_BAR, LOW); 
  //delay(1000); 
  SPI.transfer16(bitstream); 
  digitalWrite(PIN_LIN_POT_RR_CS_BAR, HIGH);
  bitstream = 0; 
  bitstream = (bitstream<<4) + 1; 
  bitstream = (bitstream<<2) + LIN_POT_WRITE_COMMAND; 
  bitstream = bitstream<<2; 
  //bitstream = (bitstream<<2) + 1; //Enable wiper
  bitstream = (bitstream<<8) + 256; 
  //bitstream = (bitstream<<10) + data; 
  digitalWrite(PIN_LIN_POT_RR_CS_BAR, LOW); 
  //delay(1000); 
  SPI.transfer16(bitstream); 
  digitalWrite(PIN_LIN_POT_RR_CS_BAR, HIGH);
}

void flushBuffer() {
  while (Serial.available()) { // Flush buffer
    Serial.read(); 
  }
  readBufferIndex = 0; 
}



void loop() {
  static unsigned long timeAtRead = 0; 
  while (Serial.available() && readBufferIndex < READ_BUFFER_SIZE) {
    readBuffer[readBufferIndex] = Serial.read(); 
    readBufferIndex++; 
    timeAtRead = millis(); 
  }
  switch (mode) {
    case MODE_INIT: modeInit(); break; 
    case MODE_BUILD: modeBuild(); break; 
    case MODE_RUN: modeRun(); break; 
    case MODE_DEBUG: modeDebug(); break; 
    default: modeUnknown(); break; 
  }
  if (readBufferIndex > 0 && (millis() - timeAtRead) > 500) {
    // Fallback situation. Return invalid request if incomplete request is held for 1 sec
    Serial.println(F("Time limit exceeded")); 
    for (int i = 0; i < readBufferIndex; i++) {
      Serial.print(readBuffer[i]); 
      Serial.print(F(" ")); 
    }
    Serial.println(); 
    Serial.println(SERIAL_INVALID_REQUEST); 
    flushBuffer(); 
  }
}

void modeInit() {
  if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_CHANGE_MODE)) {
    // Always enter here if you know you're going to change modes, so you don't accidently fall into the other if
    flushBuffer(); 
    changeMode(readBuffer[1]); 
  } else if ((readBufferIndex >= 1) && (readBuffer[0] == SERIAL_INIT_CHARACTERIZE)) {
    flushBuffer(); 
    initCharacterization(); 
  } else if ((readBufferIndex >= 1) && (readBuffer[0] == SERIAL_INIT_ACKNOWLEDGE)) {
    flushBuffer(); 
    initAcknowledge(); 
  } else if ((readBufferIndex >= 1) && (readBuffer[0] == SERIAL_INIT_PARALLEL_RESISTANCE)) {
    flushBuffer(); 
    initParallelResistance(); 
  } else if ((readBufferIndex >= 3) && (readBuffer[0] == SERIAL_INIT_PARALLEL_CAPACITANCE)) {
    flushBuffer(); 
    initParallelCapacitance(readBuffer[1], readBuffer[2]); 
  }
}

void changeMode(int newMode) {
  if (newMode == mode) {
    Serial.print(F("Arduino state is already in requested mode: ")); 
    Serial.println(mode); 
    Serial.println(SERIAL_VALID_REQUEST); 
    return; 
  }
  switch (newMode) {
    case MODE_INIT:
    case MODE_BUILD:
    case MODE_RUN:
    case MODE_DEBUG:
      Serial.print(F("Changing mode to: ")); 
      Serial.println(newMode); 
      Serial.println(SERIAL_VALID_REQUEST); 
      digitalWrite(PIN_CIRCUIT_POWER, LOW); 
      digitalWrite(PIN_DEMUX_ENABLE, LOW); 
      for (int i = 0; i < PIN_DEMUX_SELECT_LENGTH; i++) {
        digitalWrite(PIN_DEMUX_SELECT_START+i, LOW); 
      }
      digitalWrite(PIN_MUX_ENABLE, LOW); 
      // MUX REWIRING
      //for (int i = 0; i < PIN_MUX_SELECT_LENGTH; i++) {
      //  digitalWrite(PIN_MUX_SELECT_START+i, LOW); 
      //}
      setMuxSelectMuxRewired(0); 
      digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, LOW); 
      for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
        digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+i, LOW); 
      }
      if (newMode == MODE_BUILD) {
        changeReadRail(0); 
      } else if (newMode == MODE_RUN) {
        changeReadRail(PIN_RAIL_READ_LENGTH-1); 
      } else {
        changeReadRail(PIN_RAIL_READ_LENGTH-1); 
      }
      digitalWrite(PIN_RAIL_ENABLE, LOW); 
      mode = newMode; 
      break; 
    default: 
      Serial.print(F("Invalid mode to change to: ")); 
      Serial.println(newMode); 
      Serial.println(SERIAL_INVALID_REQUEST); 
      break; 
  }
}

void initCharacterization() {
  // Measurements are calculated and reported in the following order: 
  // 1-7: 
  /* Process: 
   * Turn all mux selects to 0
   * Enable demux, mux, reference resistor mux
   * Take measurements. Sweep through all reference resistor selects to detect variance. Report
   * Take measurements. Sweep through all demux selects to detect variance. Report
   * Take measurements. Sweep through all mux selects to detect variance. Report
   * Return
   */
  Serial.println(F("Generating reference measurements")); 
  Serial.println(SERIAL_VALID_REQUEST); 
  digitalWrite(PIN_RAIL_ENABLE, HIGH); 
  changeReadRail(0); 
  for (int i = 0; i < PIN_DEMUX_SELECT_LENGTH; i++) {
    digitalWrite(PIN_DEMUX_SELECT_START+i, LOW); 
  }
  // MUX REWIRING
  setMuxSelectMuxRewired(0); 
  //for (int i = 0; i < PIN_MUX_SELECT_LENGTH; i++) {
  //  digitalWrite(PIN_MUX_SELECT_START+i, LOW); 
  //}
  for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
    digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+i, LOW); 
  }
  digitalWrite(PIN_DEMUX_ENABLE, HIGH); 
  digitalWrite(PIN_MUX_ENABLE, HIGH); 
  digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, HIGH); 
  unsigned int voltageReadRail; 
  unsigned int voltageReadResistor; 
  float averageVoltageReadRail; 
  float averageVoltageReadResistor; 
  for (int i = 0; i < REFERENCE_RESISTOR_SELECT_SIZE; i++) {
    voltageReadRail = 0; 
    voltageReadResistor = 0; 
    // Select the next rail
    for (int ii = 0; ii < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; ii++) {
      digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+ii, (1 << ii) & i); 
    }
    // Eliminate possible time variance
    delayMicroseconds(INIT_SWITCH_DELAY_US); 
    // Take measurements and average
    for (int ii = 0; ii < INIT_NUM_MEASUREMENTS; ii++) {
      voltageReadRail += analogRead(PIN_RAIL_READ_START+readRailSelected); 
      //voltageReadRail += analogRead(PIN_RAIL_ANALOG_READ); 
      voltageReadResistor += analogRead(PIN_REFERENCE_RESISTOR_ANALOG_READ_START+i); 
      delayMicroseconds(INIT_READ_DELAY_US); 
    }
    // Use floats to prevent rounding
    averageVoltageReadRail = (float)voltageReadRail/INIT_NUM_MEASUREMENTS; 
    averageVoltageReadResistor = (float)voltageReadResistor/INIT_NUM_MEASUREMENTS; 
    // Send the raw data and let Python interpret it
    // Data is sent by default with 2 decimal points. That should be acceptable
    Serial.println(averageVoltageReadRail); 
    Serial.println(averageVoltageReadResistor); 
  }
  for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
      digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+i, LOW); 
  }
  for (int i = 0; i < MUX_SELECT_SIZE; i++) {
    voltageReadRail = 0; 
    voltageReadResistor = 0; 
    // Select the next rail
    for (int ii = 0; ii < PIN_DEMUX_SELECT_LENGTH; ii++) {
      digitalWrite(PIN_DEMUX_SELECT_START+ii, (1 << ii) & i); 
    }
    // MUX REWIRING CHANGE
    //for (int ii = 0; ii < PIN_MUX_SELECT_LENGTH; ii++) {
    //  digitalWrite(PIN_MUX_SELECT_START+ii, (1 << ii) & i); 
    //}
    setMuxSelectMuxRewired(i); 
    // Eliminate possible time variance
    delayMicroseconds(INIT_SWITCH_DELAY_US); 
    // Take measurements and average
    for (int ii = 0; ii < INIT_NUM_MEASUREMENTS; ii++) {
      // voltageReadRail += analogRead(PIN_RAIL_ANALOG_READ); 
      voltageReadRail += analogRead(PIN_RAIL_READ_START+readRailSelected); 
      voltageReadResistor += analogRead(PIN_REFERENCE_RESISTOR_ANALOG_READ_START); 
      delayMicroseconds(INIT_READ_DELAY_US); 
    }
    averageVoltageReadRail = voltageReadRail/INIT_NUM_MEASUREMENTS; 
    averageVoltageReadResistor = voltageReadResistor/INIT_NUM_MEASUREMENTS; 
    Serial.println(averageVoltageReadRail); 
    Serial.println(averageVoltageReadResistor); 
  }
  for (int i = 0; i < PIN_DEMUX_SELECT_LENGTH; i++) {
    digitalWrite(PIN_DEMUX_SELECT_START+i, LOW); 
  }
  // MUX REWIRING
  setMuxSelectMuxRewired(0); 
  //for (int i = 0; i < PIN_MUX_SELECT_LENGTH; i++) {
  //  digitalWrite(PIN_MUX_SELECT_START+i, LOW); 
  //}
  digitalWrite(PIN_DEMUX_ENABLE, LOW); 
  digitalWrite(PIN_MUX_ENABLE, LOW); 
  digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, LOW); 
  digitalWrite(PIN_RAIL_ENABLE, LOW); 
}

void initAcknowledge() {
  Serial.println("Notice me Senpai"); 
  Serial.println(SERIAL_VALID_REQUEST); 
}

// Calculate parallel resistance
// Note that we make the assumption that resistances here are only decoupled with a single mux or demux rail
// This eliminates the need to check 64*64 pairs each for demux and mux, and instead check just 64*2 pairings
// Note also that because we're only checking 2 pairs, we assume that the odds of any pairing having relevant resistance are small, 
//   so it is unlikely to have a single rail accidently pair with 2 adjacent rails having high resistance
void initParallelResistance() {
  Serial.println(F("Generating parallel resistance measurements")); 
  Serial.println(SERIAL_VALID_REQUEST); 
  digitalWrite(PIN_RAIL_ENABLE, HIGH); 
  changeReadRail(0); 
  setMuxSelect(PIN_REFERENCE_RESISTOR_SELECT_START, PIN_REFERENCE_RESISTOR_SELECT_LENGTH, REFERENCE_RESISTOR_SELECT_SIZE-1); 
  setMuxSelect(PIN_DEMUX_SELECT_START, PIN_DEMUX_SELECT_LENGTH, 0); 
  // MUX REWIRING
  //setMuxSelect(PIN_MUX_SELECT_START, PIN_MUX_SELECT_LENGTH, 1); 
  setMuxSelectMuxRewired(1); 
  digitalWrite(PIN_DEMUX_ENABLE, HIGH); 
  digitalWrite(PIN_MUX_ENABLE, HIGH); 
  digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, HIGH); 
  for (int i = 0; i < MUX_SELECT_SIZE; i++) {
    setMuxSelect(PIN_DEMUX_SELECT_START, PIN_DEMUX_SELECT_LENGTH, i); 
    // MUX REWIRING
    //setMuxSelect(PIN_MUX_SELECT_START, PIN_MUX_SELECT_LENGTH, (i+1)%MUX_SELECT_SIZE); 
    setMuxSelectMuxRewired((i+1)%MUX_SELECT_SIZE); 
    int voltageReadRail = 0; 
    for (int ii = 0; ii < INIT_NUM_MEASUREMENTS; ii++) {
      voltageReadRail += analogRead(PIN_RAIL_READ_START+readRailSelected); 
      delay(1); 
    }
    int averageVoltageReadRail = voltageReadRail/INIT_NUM_MEASUREMENTS; 
    Serial.println(averageVoltageReadRail); 
  }
  for (int i = 0; i < MUX_SELECT_SIZE; i++) {
    setMuxSelect(PIN_DEMUX_SELECT_START, PIN_DEMUX_SELECT_LENGTH, i); 
    // MUX REWIRING
    //setMuxSelect(PIN_MUX_SELECT_START, PIN_MUX_SELECT_LENGTH, (i-1+64)%MUX_SELECT_SIZE); 
    setMuxSelectMuxRewired((i-1+64)%MUX_SELECT_SIZE); 
    int voltageReadRail = 0; 
    for (int ii = 0; ii < INIT_NUM_MEASUREMENTS; ii++) {
      voltageReadRail += analogRead(PIN_RAIL_READ_START+readRailSelected); 
      delay(1); 
    }
    int averageVoltageReadRail = voltageReadRail/INIT_NUM_MEASUREMENTS; 
    Serial.println(averageVoltageReadRail); 
  }
  digitalWrite(PIN_DEMUX_ENABLE, LOW); 
  digitalWrite(PIN_MUX_ENABLE, LOW); 
  digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, LOW); 
  setMuxSelect(PIN_REFERENCE_RESISTOR_SELECT_START, PIN_REFERENCE_RESISTOR_SELECT_LENGTH, referenceResistorRailSelected); 
  setMuxSelect(PIN_DEMUX_SELECT_START, PIN_DEMUX_SELECT_LENGTH, 0); 
  // MUX REWIRING
  //setMuxSelect(PIN_MUX_SELECT_START, PIN_MUX_SELECT_LENGTH, 0); 
  setMuxSelectMuxRewired(0); 
}

// Calculate parallel capacitance
// Note that we make the assumption that capacitances here are only decoupled with a single mux or demux rail
// This eliminates the need to check 64*64 pairs each for demux and mux, and instead check just 64*2 pairings
// Note also that because we're only checking 2 pairs, we assume that the odds of any pairing having relevant resistance are small, 
//   so it is unlikely to have a single rail accidently pair with 2 adjacent rails having high capacitance
void initParallelCapacitance(int potHigh, int potLow) {
  Serial.println(F("Generating parallel capacitance measurements")); 
  Serial.println(SERIAL_VALID_REQUEST); 
  digitalWrite(PIN_RAIL_ENABLE, HIGH); 
  changeReadRail(0); 
  changeLinPot(0, potHigh); 
  changeLinPot(1, potLow); 
  setMuxSelect(PIN_REFERENCE_RESISTOR_SELECT_START, PIN_REFERENCE_RESISTOR_SELECT_LENGTH, REFERENCE_RESISTOR_SELECT_SIZE-1); 
  setMuxSelect(PIN_DEMUX_SELECT_START, PIN_DEMUX_SELECT_LENGTH, 0); 
  // MUX REWIRING
  //setMuxSelect(PIN_MUX_SELECT_START, PIN_MUX_SELECT_LENGTH, 1); 
  setMuxSelectMuxRewired(1); 
  digitalWrite(PIN_DEMUX_ENABLE, HIGH); 
  digitalWrite(PIN_MUX_ENABLE, HIGH); 
  //digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, HIGH); 
  int interruptTrigger; 
  unsigned long timeReference; 
  unsigned long timeSample; 
  for (int i = 0; i < MUX_SELECT_SIZE; i++) {
    setMuxSelect(PIN_DEMUX_SELECT_START, PIN_DEMUX_SELECT_LENGTH, i); 
    // MUX REWIRING
    //setMuxSelect(PIN_MUX_SELECT_START, PIN_MUX_SELECT_LENGTH, (i+1)%MUX_SELECT_SIZE); 
    setMuxSelectMuxRewired((i+1)%MUX_SELECT_SIZE); 
    attachInterrupt(digitalPinToInterrupt(PIN_COMP_INTERRUPT), ISRTime, FALLING); 
    interruptTrigger = 0; 
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, HIGH); 
    timeReference = micros(); 
    while (interruptTrigger == 0) {
      Serial.print(F("DEBUG: ")); 
      Serial.println(analogRead(PIN_RAIL_READ_START+readRailSelected)); 
      //delay(1); 
    }
    detachInterrupt(digitalPinToInterrupt(PIN_COMP_INTERRUPT)); 
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, LOW); 
    timeSample = interruptTime - timeReference; 
    Serial.println(timeSample); 
  }
  for (int i = 0; i < MUX_SELECT_SIZE; i++) {
    setMuxSelect(PIN_DEMUX_SELECT_START, PIN_DEMUX_SELECT_LENGTH, i); 
    // MUX REWIRING
    //setMuxSelect(PIN_MUX_SELECT_START, PIN_MUX_SELECT_LENGTH, (i-1+64)%MUX_SELECT_SIZE); 
    setMuxSelectMuxRewired((i-1+64)%MUX_SELECT_SIZE); 
    attachInterrupt(digitalPinToInterrupt(PIN_COMP_INTERRUPT), ISRTime, FALLING); 
    interruptTrigger = 0; 
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, HIGH); 
    timeReference = micros(); 
    while (interruptTrigger == 0) {
      Serial.print(F("DEBUG: ")); 
      Serial.println(analogRead(PIN_RAIL_READ_START+readRailSelected)); 
      //delay(1); 
    }
    detachInterrupt(digitalPinToInterrupt(PIN_COMP_INTERRUPT)); 
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, LOW); 
    timeSample = interruptTime - timeReference; 
    Serial.println(timeSample); 
  }
  digitalWrite(PIN_DEMUX_ENABLE, LOW); 
  digitalWrite(PIN_MUX_ENABLE, LOW); 
  digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, LOW); 
  setMuxSelect(PIN_REFERENCE_RESISTOR_SELECT_START, PIN_REFERENCE_RESISTOR_SELECT_LENGTH, referenceResistorRailSelected); 
  setMuxSelect(PIN_DEMUX_SELECT_START, PIN_DEMUX_SELECT_LENGTH, 0); 
  // MUX REWIRING
  //setMuxSelect(PIN_MUX_SELECT_START, PIN_MUX_SELECT_LENGTH, 0); 
  setMuxSelectMuxRewired(0); 
}


/* Available commands: 
 *  - Change mode
 *  - Switch reference resistor to x select
 *  - Read voltage on rail x times (resistors and diodes only)
 *  - Read voltage and time on rail x times (anything with capacitors with unknown curves)
 *  - Read time once at x voltage (anything easy with capacitors)
 */
void modeBuild() {
  if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_CHANGE_MODE)) {
    // Always enter here if you know you're going to change modes, so you don't accidently fall into the other if
    flushBuffer(); 
    changeMode(readBuffer[1]); 
  } else if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_BUILD_CHANGE_REFERENCE_RESISTOR)) {
    flushBuffer(); 
    changeReferenceResistor(readBuffer[1]); 
  } else if ((readBufferIndex >= 4) && (readBuffer[0] == SERIAL_BUILD_VOLTAGE_READ)) {
    flushBuffer(); 
    buildVoltageRead(readBuffer[1], readBuffer[2], readBuffer[3]); 
  } else if ((readBufferIndex >= 5) && (readBuffer[0] == SERIAL_BUILD_VOLTAGE_READ_DELAY)) {
    flushBuffer(); 
    buildVoltageReadDelay(readBuffer[1], readBuffer[2], readBuffer[3], readBuffer[4]); 
  } else if ((readBufferIndex >= 4) && (readBuffer[0] == SERIAL_BUILD_VOLTAGE_TIME_READ)) {
    flushBuffer(); 
    buildVoltageTimeRead(readBuffer[1], readBuffer[2], readBuffer[3]); 
  } else if ((readBufferIndex >= 7) && (readBuffer[0] == SERIAL_BUILD_TIME_READ)) {
    flushBuffer(); 
    // Highest byte sent first
    // Due to the nature of the circuit, this will stop when the analogRead falls BELOW stopVoltage
    int stopVoltage = readBuffer[3]*(1 << 8) + readBuffer[4]; 
    int maxTime = readBuffer[5]*(1 << 8) + readBuffer[6]; 
    buildTimeRead(readBuffer[1], readBuffer[2], stopVoltage, maxTime); 
  } else if ((readBufferIndex >= 3) && (readBuffer[0] == SERIAL_BUILD_DIGITAL_COMP)) {
    flushBuffer(); 
    buildDigitalComp(readBuffer[1], readBuffer[2]); 
  } else if ((readBufferIndex >= 5) && (readBuffer[0] == SERIAL_BUILD_DIGITAL_INTERRUPT)) {
    flushBuffer(); 
    buildDigitalInterrupt(readBuffer[1], readBuffer[2], readBuffer[3], readBuffer[4]); 
  }
}

void changeReferenceResistor(int rail) {
  /*
  if (resistorIndex >= 0 && resistorIndex < REFERENCE_RESISTOR_SELECT_SIZE) {
    Serial.print(F("Setting reference resistor to rail: ")); 
    Serial.println(resistorIndex); 
    Serial.println(SERIAL_VALID_REQUEST); 
    for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
      digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+i, (1 << i) & resistorIndex); 
    }
    referenceResistorRailSelected = resistorIndex; 
  } else {
    Serial.print(F("Invalid select for reference resistor mux: ")); 
    Serial.println(resistorIndex); 
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
  flushBuffer(); 
  */
  
  if ( (rail >= 0 && rail < MUX_SELECT_SIZE) || (rail == REFERENCE_RESISTOR_DRAIN_INDEX) ) {
    Serial.print(F("debugReferenceResistorSelect: Selecting rail: ")); 
    Serial.println(rail); 
    Serial.println(SERIAL_VALID_REQUEST); 
    for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
      digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+i, (1 << i) & rail); 
    }
    referenceResistorRailSelected = rail; 
  } else {
    Serial.print(F("debugReferenceResistorSelect: Invalid rail: ")); 
    Serial.println(rail); 
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

void changeReadRail(int rail) {
  if (rail >= 0 && rail < PIN_RAIL_READ_LENGTH) {
    //Serial.print(F("changeReadRail: Selecting rail: ")); 
    //Serial.println(rail); 
    //Serial.println(SERIAL_VALID_REQUEST); 
    for (int i = 0; i < PIN_RAIL_SELECT_LENGTH; i++) {
      digitalWrite(PIN_RAIL_SELECT_START+i, (1 << i) & rail); 
    }
    readRailSelected = rail; 
  } else {
    //Serial.print(F("changeReadRail: Invalid rail: ")); 
    //Serial.println(rail); 
    //Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

void buildVoltageRead(int rail1, int rail2, int numMeasurements) {
  if ( (rail1 >= 0 && rail1 < MUX_SELECT_SIZE) &&
       (rail2 >= 0 && rail2 < MUX_SELECT_SIZE) &&
       (numMeasurements >= 0) ) {
    Serial.println("buildVoltageRead: Valid request");
    Serial.println(SERIAL_VALID_REQUEST); 
    int currentReferenceResistor = 0; 
    digitalWrite(PIN_RAIL_ENABLE, HIGH); 
    changeReadRail(0); 
    for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
      currentReferenceResistor += digitalRead(PIN_REFERENCE_RESISTOR_SELECT_START+i)*(1 << i); 
    }
    for (int i = 0; i < PIN_DEMUX_SELECT_LENGTH; i++) {
      digitalWrite(PIN_DEMUX_SELECT_START+i, (1 << i) & rail1); 
    }
    // MUX REWIRING
    //for (int i = 0; i < PIN_MUX_SELECT_LENGTH; i++) {
    //  digitalWrite(PIN_MUX_SELECT_START+i, (1 << i) & rail2); 
    //}
    setMuxSelectMuxRewired(rail2); 
    digitalWrite(PIN_DEMUX_ENABLE, HIGH); 
    digitalWrite(PIN_MUX_ENABLE, HIGH); 
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, HIGH); 
    // Observation: At very high resistances (100k+) the mux doesn't work properly if it first doesn't 'discharge' after being enabled, before measuring
    // Therefore, we're going to blaze a current down one rail to discharge for whatever reason, then switch it back to the other rail and start measurements
    // MUX REWIRING
    //setMuxSelect(PIN_MUX_SELECT_START, PIN_DEMUX_SELECT_LENGTH, rail1); 
    setMuxSelectMuxRewired(rail1); 
    delay(10); 
    // MUX REWIRING
    //setMuxSelect(PIN_MUX_SELECT_START, PIN_DEMUX_SELECT_LENGTH, rail2); 
    setMuxSelectMuxRewired(rail2); 
    delay(1); 
    int voltage; 
    int voltageReferenceResistor; 
    for (int i = 0; i < numMeasurements; i++) {
      voltage = analogRead(PIN_RAIL_READ_START+readRailSelected); 
      //voltage = analogRead(PIN_RAIL_ANALOG_READ); 
      voltageReferenceResistor = analogRead(PIN_REFERENCE_RESISTOR_ANALOG_READ_START+currentReferenceResistor); 
      Serial.println(voltage); 
      Serial.println(voltageReferenceResistor); 
      delayMicroseconds(1); 
    }
    digitalWrite(PIN_DEMUX_ENABLE, LOW); 
    digitalWrite(PIN_MUX_ENABLE, LOW); 
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, LOW); 
    digitalWrite(PIN_RAIL_ENABLE, LOW); 
  } else {
    Serial.print(F("buildVoltageRead: Invalid arguments: ")); 
    Serial.print(rail1); 
    Serial.print(F(", ")); 
    Serial.print(rail2); 
    Serial.print(F(", ")); 
    Serial.println(numMeasurements); 
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

void buildVoltageReadDelay(int rail1, int rail2, int numMeasurements, int delay100ms) {
  if ( (rail1 >= 0 && rail1 < MUX_SELECT_SIZE) &&
       (rail2 >= 0 && rail2 < MUX_SELECT_SIZE) &&
       (numMeasurements >= 0) ) {
    Serial.println("buildVoltageReadDelay: Valid request");
    Serial.println(SERIAL_VALID_REQUEST); 
    digitalWrite(PIN_RAIL_ENABLE, HIGH); 
    changeReadRail(0); 
    int currentReferenceResistor = 0; 
    for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
      currentReferenceResistor += digitalRead(PIN_REFERENCE_RESISTOR_SELECT_START+i)*(1 << i); 
    }
    for (int i = 0; i < PIN_DEMUX_SELECT_LENGTH; i++) {
      digitalWrite(PIN_DEMUX_SELECT_START+i, (1 << i) & rail1); 
    }
    // MUX REWIRING
    //for (int i = 0; i < PIN_MUX_SELECT_LENGTH; i++) {
    //  digitalWrite(PIN_MUX_SELECT_START+i, (1 << i) & rail2); 
    //}
    setMuxSelectMuxRewired(rail2); 
    digitalWrite(PIN_DEMUX_ENABLE, HIGH); 
    digitalWrite(PIN_MUX_ENABLE, HIGH); 
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, HIGH); 
    // Observation: At very high resistances (100k+) the mux doesn't work properly if it first doesn't 'discharge' after being enabled, before measuring
    // Therefore, we're going to blaze a current down one rail to discharge for whatever reason, then switch it back to the other rail and start measurements
    // MUX REWIRING
    //setMuxSelect(PIN_MUX_SELECT_START, PIN_DEMUX_SELECT_LENGTH, rail1); 
    setMuxSelectMuxRewired(rail1); 
    delay(10); 
    // MUX REWIRING
    //setMuxSelect(PIN_MUX_SELECT_START, PIN_DEMUX_SELECT_LENGTH, rail2);
    setMuxSelectMuxRewired(rail2); 
    delay(delay100ms*100); 
    int voltage; 
    int voltageReferenceResistor; 
    for (int i = 0; i < numMeasurements; i++) {
      // voltage = analogRead(PIN_RAIL_ANALOG_READ); 
      voltage = analogRead(PIN_RAIL_READ_START+readRailSelected); 
      voltageReferenceResistor = analogRead(PIN_REFERENCE_RESISTOR_ANALOG_READ_START+currentReferenceResistor); 
      Serial.println(voltage); 
      Serial.println(voltageReferenceResistor); 
      delayMicroseconds(1); 
    }
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, LOW); 
    // Swap reference resistor to the drain channel
    for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
      digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+i, (1 << i) & REFERENCE_RESISTOR_DRAIN_INDEX); 
    }
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, HIGH); 
    delay(delay100ms*100); 
    digitalWrite(PIN_DEMUX_ENABLE, LOW); 
    digitalWrite(PIN_MUX_ENABLE, LOW); 
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, LOW); 
    // Return reference resistor back to original
    for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
      digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+i, (1 << i) & referenceResistorRailSelected); 
    }
    digitalWrite(PIN_RAIL_ENABLE, LOW); 
  } else {
    Serial.print(F("buildVoltageRead: Invalid arguments: ")); 
    Serial.print(rail1); 
    Serial.print(F(", ")); 
    Serial.print(rail2); 
    Serial.print(F(", ")); 
    Serial.println(numMeasurements); 
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

void buildVoltageTimeRead(int rail1, int rail2, int numMeasurements) {
  if ( (rail1 >= 0 && rail1 < MUX_SELECT_SIZE) &&
       (rail2 >= 0 && rail2 < MUX_SELECT_SIZE) &&
       (numMeasurements >= 0) ) {
    Serial.println(F("buildVoltageTimeRead: Valid request"));
    Serial.println(SERIAL_VALID_REQUEST); 
    digitalWrite(PIN_RAIL_ENABLE, HIGH); 
    changeReadRail(0);
    for (int i = 0; i < PIN_DEMUX_SELECT_LENGTH; i++) {
      digitalWrite(PIN_DEMUX_SELECT_START+i, (1 << i) & rail1); 
    }
    // MUX REWIRING
    //for (int i = 0; i < PIN_MUX_SELECT_LENGTH; i++) {
    //  digitalWrite(PIN_MUX_SELECT_START+i, (1 << i) & rail2); 
    //}
    setMuxSelectMuxRewired(rail2); 
    int voltageSample; 
    unsigned long timeSample; 
    unsigned long timeReference; 
    unsigned long initDrainTime = 100000; 
    digitalWrite(PIN_DEMUX_ENABLE, HIGH); 
    digitalWrite(PIN_MUX_ENABLE, HIGH); 
    for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
      digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+i, (1 << i) & REFERENCE_RESISTOR_DRAIN_INDEX); 
    }
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, HIGH); 
    timeReference = micros(); 
    // Allow capacitor to drain for an equal amount of time it was charging
    while ((micros() - timeReference) < initDrainTime) {}
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, LOW); 
    // Return reference resistor back to original
    for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
      digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+i, (1 << i) & referenceResistorRailSelected); 
    }
    // Observation: At very high resistances (100k+) the mux doesn't work properly if it first doesn't 'discharge' after being enabled, before measuring
    // Therefore, we're going to blaze a current down one rail to discharge for whatever reason, then switch it back to the other rail and start measurements
    ////setMuxSelect(PIN_MUX_SELECT_START, PIN_DEMUX_SELECT_LENGTH, rail1); 
    ////MUX REWIRING
    //setMuxSelectMuxRewired(rail1); 
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, HIGH); 
    //delay(10); 
    ////setMuxSelect(PIN_MUX_SELECT_START, PIN_DEMUX_SELECT_LENGTH, rail2);
    //// MUX REWIRING
    //setMuxSelectMuxRewired(rail2); 
    timeReference = micros(); 
    for (int i = 0; i < numMeasurements; i++) {
      voltageSample = analogRead(PIN_RAIL_READ_START+readRailSelected); 
      // voltageSample = analogRead(PIN_RAIL_ANALOG_READ); 
      timeSample = micros() - timeReference; 
      Serial.println(timeSample); 
      Serial.println(voltageSample); 
      delayMicroseconds(1); 
    }
    //Serial.println("FINISH"); 
    // Drain possible capacitor
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, LOW); 
    // Swap reference resistor to the drain channel
    for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
      digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+i, (1 << i) & REFERENCE_RESISTOR_DRAIN_INDEX); 
    }
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, HIGH); 
    unsigned long timeReference2 = micros(); 
    // Allow capacitor to drain for an equal amount of time it was charging
    while ((micros() - timeReference2) < (timeReference2 - timeReference)*2) {}
    digitalWrite(PIN_DEMUX_ENABLE, LOW); 
    digitalWrite(PIN_MUX_ENABLE, LOW); 
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, LOW); 
    // Return reference resistor back to original
    for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
      digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+i, (1 << i) & referenceResistorRailSelected); 
    }
    digitalWrite(PIN_RAIL_ENABLE, LOW); 
  } else {
    Serial.print(F("buildTimeRead: Invalid arguments: ")); 
    Serial.print(rail1); 
    Serial.print(F(", ")); 
    Serial.print(rail2); 
    Serial.print(F(", ")); 
    Serial.println(numMeasurements); 
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

// TODO: change this to remove the arrays because they are limiting the work space
void buildTimeRead(int rail1, int rail2, int stopVoltage, unsigned long maxTime) {
  if ( (rail1 >= 0 && rail1 < MUX_SELECT_SIZE) &&
       (rail2 >= 0 && rail2 < MUX_SELECT_SIZE) ) {
    Serial.println(F("buildeTimeRead: Valid request"));
    Serial.println(SERIAL_VALID_REQUEST); 
    digitalWrite(PIN_RAIL_ENABLE, HIGH); 
    changeReadRail(0); 
    for (int i = 0; i < PIN_DEMUX_SELECT_LENGTH; i++) {
      digitalWrite(PIN_DEMUX_SELECT_START+i, (1 << i) & rail1); 
    }
    // MUX REWIRING
    //for (int i = 0; i < PIN_MUX_SELECT_LENGTH; i++) {
    //  digitalWrite(PIN_MUX_SELECT_START+i, (1 << i) & rail2); 
    //}
    setMuxSelectMuxRewired(rail2); 
    unsigned long timeReference; 
    unsigned long timeSample; 
    unsigned long initDrainTime = 100000; 
    // maxTime is provided in ms, but we will measure in us
    maxTime = maxTime * 1000; 
    digitalWrite(PIN_DEMUX_ENABLE, HIGH); 
    digitalWrite(PIN_MUX_ENABLE, HIGH); 
    // Ensure capacitor is drained
    for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
      digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+i, (1 << i) & REFERENCE_RESISTOR_DRAIN_INDEX); 
    }
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, HIGH); 
    timeReference = micros(); 
    while (micros() - timeReference < initDrainTime) {}
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, LOW); 
    // Return reference resistor back to original
    for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
      digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+i, (1 << i) & referenceResistorRailSelected); 
    }
    timeReference = micros(); 
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, HIGH); 
    //while ( (analogRead(PIN_RAIL_ANALOG_READ) > stopVoltage) && 
    //        (micros()-timeReference < maxTime) ) {}
    while ( (analogRead(PIN_RAIL_READ_START+readRailSelected) > stopVoltage) && 
            (micros() - timeReference < maxTime) ) {}
    timeSample = micros() - timeReference; 
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, LOW); 
    // Swap reference resistor to the drain channel
    for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
      digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+i, (1 << i) & REFERENCE_RESISTOR_DRAIN_INDEX); 
    }
    // Observation: At very high resistances (100k+) the mux doesn't work properly if it first doesn't 'discharge' after being enabled, before measuring
    // Therefore, we're going to blaze a current down one rail to discharge for whatever reason, then switch it back to the other rail and start measurements
    ////MUX REWIRING
    // setMuxSelectMuxRewired(rail1); 
    ////setMuxSelect(PIN_MUX_SELECT_START, PIN_DEMUX_SELECT_LENGTH, rail1); 
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, HIGH); 
    //delay(10); 
    ////MUX REWIRING
    //setMuxSelectMuxRewired(rail2); 
    ////setMuxSelect(PIN_MUX_SELECT_START, PIN_DEMUX_SELECT_LENGTH, rail2);
    timeReference = micros(); 
    // Allow capacitor to drain for an equal amount of time it was charging
    while ((micros() - timeReference) < timeSample*2) {}
    digitalWrite(PIN_DEMUX_ENABLE, LOW); 
    digitalWrite(PIN_MUX_ENABLE, LOW); 
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, LOW); 
    // Return reference resistor back to original
    for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
      digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+i, (1 << i) & referenceResistorRailSelected); 
    }
    digitalWrite(PIN_RAIL_ENABLE, LOW); 
    Serial.println(timeSample); 
  } else {
    Serial.print(F("buildTimeRead: Invalid arguments: ")); 
    Serial.print(rail1); 
    Serial.print(F(", ")); 
    Serial.print(rail2); 
    Serial.print(F(", ")); 
    Serial.print(stopVoltage); 
    Serial.print(F(", ")); 
    Serial.println(maxTime); 
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

void buildDigitalComp(int rail1, int rail2) {
  if ( (rail1 >= 0 && rail1 < MUX_SELECT_SIZE) &&
       (rail2 >= 0 && rail2 < MUX_SELECT_SIZE) ) {
    Serial.println(F("buildDigitalComp: Valid request"));
    Serial.println(SERIAL_VALID_REQUEST); 
    for (int i = 0; i < PIN_DEMUX_SELECT_LENGTH; i++) {
      digitalWrite(PIN_DEMUX_SELECT_START+i, (1 << i) & rail1); 
    }
    // MUX REWIRING
    //for (int i = 0; i < PIN_MUX_SELECT_LENGTH; i++) {
    //  digitalWrite(PIN_MUX_SELECT_START+i, (1 << i) & rail2); 
    //}
    setMuxSelectMuxRewired(rail2); 
    for (int i = 0; i < PIN_MUX_SELECT_LENGTH; i++) {
      digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+i, (1 << i) & REFERENCE_RESISTOR_POT_INDEX); 
    }
    int pot0 = 256; //Upper leg
    int pot1 = 256; //Lower leg
    int potRR = 256; 
    int pot0LowRange = 0;
    int pot0HighRange = 256; 
    int pot1LowRange = 0;
    int pot1HighRange = 256; 
    int potRRLowRange = 0;
    int potRRHighRange = 256; 
    changeLinPot(0, pot0); 
    changeLinPot(1, pot1); 
    changeLinPotRR(1, potRR); 
    //bool continueMeasurement = 1; 
    digitalWrite(PIN_DEMUX_ENABLE, HIGH); 
    digitalWrite(PIN_MUX_ENABLE, HIGH); 
    // Observation: At very high resistances (100k+) the mux doesn't work properly if it first doesn't 'discharge' after being enabled, before measuring
    // Therefore, we're going to blaze a current down one rail to discharge for whatever reason, then switch it back to the other rail and start measurements
    // MUX REWIRING
    //setMuxSelect(PIN_MUX_SELECT_START, PIN_DEMUX_SELECT_LENGTH, rail1); 
    setMuxSelectMuxRewired(rail1); 
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, HIGH); 
    delay(10); 
    // MUX REWIRING
    //setMuxSelect(PIN_MUX_SELECT_START, PIN_DEMUX_SELECT_LENGTH, rail2);
    setMuxSelectMuxRewired(rail2); 
    int lastRead = digitalRead(PIN_COMP_READ); 
    int lastPotChanged = -1; 
    int lastPotDelta = 0; 
    // Naive algorithm. Better later to use 'best rational approximation', Farey sequence
    if (digitalRead(PIN_COMP_READ) == HIGH) {
      while ( (digitalRead(PIN_COMP_READ) == HIGH) && (pot0 > 0 || potRR > 0) ) {
        if (potRR >= pot0) {
          potRR--; 
          changeLinPotRR(1, potRR); 
          lastPotChanged = 2; 
          lastPotDelta = -1; 
        } else {
          pot0--; 
          changeLinPot(0, pot0); 
          lastPotChanged = 0; 
          lastPotDelta = -1; 
        }
        Serial.print(F("DEBUG: ")); 
        Serial.print(pot0); 
        Serial.print(F(", ")); 
        Serial.print(pot1); 
        Serial.print(F(", ")); 
        Serial.println(potRR); 
        delay(100); 
      }
    } else {
      while ( (digitalRead(PIN_COMP_READ) == LOW) && (pot1 > 0) ) {
        pot1--; 
        changeLinPot(1, pot1); 
        lastPotChanged = 1; 
        lastPotDelta = -1; 
        Serial.print(F("DEBUG: ")); 
        Serial.print(pot0); 
        Serial.print(F(", ")); 
        Serial.print(pot1); 
        Serial.print(F(", ")); 
        Serial.println(potRR);
      delay(100); 
      }
    }
    digitalWrite(PIN_DEMUX_ENABLE, LOW); 
    digitalWrite(PIN_MUX_ENABLE, LOW); 
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, LOW); 
    // Return reference resistor back to original
    for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
      digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+i, (1 << i) & referenceResistorRailSelected); 
    }
    changeLinPot(0, 256); 
    changeLinPot(1, 256); 
    changeLinPotRR(1, 256); 
    Serial.println(pot0); 
    Serial.println(pot1); 
    Serial.println(potRR); 
    switch (lastPotChanged) {
      case 0: pot0 -= lastPotDelta; break; 
      case 1: pot1 -= lastPotDelta; break; 
      case 2: potRR -= lastPotDelta; break; 
    }
    Serial.println(pot0); 
    Serial.println(pot1); 
    Serial.println(potRR); 
  } else {
    Serial.print(F("buildDigitalComp: Invalid arguments: ")); 
    Serial.print(rail1); 
    Serial.print(F(", ")); 
    Serial.println(rail2); 
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

void buildDigitalInterrupt(int rail1, int rail2, int potHigh, int potLow) {
  if ( (rail1 >= 0 && rail1 < MUX_SELECT_SIZE) &&
       (rail2 >= 0 && rail2 < MUX_SELECT_SIZE) && 
       (potHigh >= 0 && potHigh < 257) && 
       (potLow >= 0 && potLow < 257) ) {
    Serial.println(F("buildDigitalInterrupt: Valid request"));
    Serial.println(SERIAL_VALID_REQUEST); 
    for (int i = 0; i < PIN_DEMUX_SELECT_LENGTH; i++) {
      digitalWrite(PIN_DEMUX_SELECT_START+i, (1 << i) & rail1); 
    }
    // MUX REWIRING
    //for (int i = 0; i < PIN_MUX_SELECT_LENGTH; i++) {
    //  digitalWrite(PIN_MUX_SELECT_START+i, (1 << i) & rail2); 
    //}
    setMuxSelectMuxRewired(rail2); 
    changeLinPot(0, potHigh); 
    changeLinPot(1, potLow); 
    // DEBUG: Test to ensure that pots are at correct resistances noted in python
    //delay(20000); 

    //// MUX REWIRING
    //setMuxSelectMuxRewired(REFERENCE_RESISTOR_POT_INDEX); 
    ////for (int i = 0; i < PIN_MUX_SELECT_LENGTH; i++) {
    ////  digitalWrite(PIN_MUX_SELECT_START+i, (1 << i) & REFERENCE_RESISTOR_POT_INDEX); 
    ////}
    unsigned long timeReference; 
    unsigned long timeSample; 
    unsigned long initDrainTime = 100000; 
    // maxTime is provided in ms, but we will measure in us
    //maxTime = maxTime * 1000; 
    digitalWrite(PIN_DEMUX_ENABLE, HIGH); 
    digitalWrite(PIN_MUX_ENABLE, HIGH); 
    // Ensure capacitor is drained
    for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
      digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+i, (1 << i) & REFERENCE_RESISTOR_DRAIN_INDEX); 
    }
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, HIGH); 
    timeReference = micros(); 
    while (micros() - timeReference < initDrainTime) {}
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, LOW); 
    // Return reference resistor back to original
    for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
      digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+i, (1 << i) & referenceResistorRailSelected); 
    }
    /*
    attachInterrupt(digitalPinToInterrupt(PIN_COMP_INTERRUPT), ISRTime, FALLING); 
    interruptTrigger = 0; 
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, HIGH); 
    timeReference = micros(); 
    while (interruptTrigger == 0) {
      Serial.print(F("DEBUG: ")); 
      Serial.println(analogRead(PIN_RAIL_READ_START+readRailSelected)); 
      //delay(1); 
    }
    detachInterrupt(digitalPinToInterrupt(PIN_COMP_INTERRUPT)); 
    timeSample = interruptTime - timeReference; 
    */
    ///*
    // Observation: At very high resistances (100k+) the mux doesn't work properly if it first doesn't 'discharge' after being enabled, before measuring
    // Therefore, we're going to blaze a current down one rail to discharge for whatever reason, then switch it back to the other rail and start measurements
    // MUX REWIRING
    //setMuxSelect(PIN_MUX_SELECT_START, PIN_DEMUX_SELECT_LENGTH, rail1); 
    setMuxSelectMuxRewired(rail1); 
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, HIGH); 
    delay(10); 
    // MUX REWIRING
    //setMuxSelect(PIN_MUX_SELECT_START, PIN_DEMUX_SELECT_LENGTH, rail2);
    setMuxSelectMuxRewired(rail2); 
    timeReference = micros(); 
    while (digitalRead(PIN_COMP_READ) == HIGH) {
      //Serial.print(F("DEBUG: ")); 
      //Serial.println(analogRead(PIN_RAIL_ANALOG_READ)); 
      //Serial.print(F("DEBUG: ")); 
      //Serial.println(micros()-timeReference); 
    }
    timeSample = micros() - timeReference; 
    //*/
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, LOW); 
    // Swap reference resistor to the drain channel
    for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
      digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+i, (1 << i) & REFERENCE_RESISTOR_DRAIN_INDEX); 
    }
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, HIGH); 
    timeReference = micros(); 
    // Allow capacitor to drain for an equal amount of time it was charging
    while ((micros() - timeReference) < timeSample*2) {}
    digitalWrite(PIN_DEMUX_ENABLE, LOW); 
    digitalWrite(PIN_MUX_ENABLE, LOW); 
    digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, LOW); 
    // Return reference resistor back to original
    for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
      digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+i, (1 << i) & referenceResistorRailSelected); 
    }
    changeLinPot(0, 256); 
    changeLinPot(1, 256); 
    Serial.println(timeSample); 
  } else {
    Serial.print(F("buildDigitalInterrupt: Invalid arguments: ")); 
    Serial.print(rail1); 
    Serial.print(F(", ")); 
    Serial.print(rail2); 
    Serial.print(F(", ")); 
    Serial.print(potHigh); 
    Serial.print(F(", ")); 
    Serial.println(potLow); 
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

void ISRTime() {
  interruptTime = micros(); 
  interruptTrigger = 1; 
}

void changeLinPot(unsigned int potNum, unsigned int data) {
  if (potNum < 2 && data < 257) {
    unsigned int bitstream = 0; 
    bitstream = (bitstream<<4) + potNum; 
    bitstream = (bitstream<<2) + LIN_POT_WRITE_COMMAND; 
    bitstream = bitstream<<2; 
    //bitstream = (bitstream<<2) + 1; //Enable wiper
    bitstream = (bitstream<<8) + data; 
    //bitstream = (bitstream<<10) + data; 
    digitalWrite(PIN_LIN_POT_CS_BAR, LOW); 
    //delay(1000); 
    SPI.transfer16(bitstream); 
    digitalWrite(PIN_LIN_POT_CS_BAR, HIGH); 
    //delay(1000); 
  }
}

void changeLinPotRR(unsigned int potNum, unsigned int data) {
  if (potNum < 2 && data < 257) {
    unsigned int bitstream = 0; 
    bitstream = (bitstream<<4) + potNum; 
    bitstream = (bitstream<<2) + LIN_POT_WRITE_COMMAND; 
    bitstream = bitstream<<2; 
    //bitstream = (bitstream<<2) + 1; //Enable wiper
    bitstream = (bitstream<<8) + data; 
    //bitstream = (bitstream<<10) + data; 
    digitalWrite(PIN_LIN_POT_RR_CS_BAR, LOW); 
    //delay(1000); 
    SPI.transfer16(bitstream); 
    digitalWrite(PIN_LIN_POT_RR_CS_BAR, HIGH); 
    //delay(1000); 
  }
}


/* Available commands: 
 *  - Change mode
 *  - Toggle circuit power
 *  - Adjust circuit voltage? 
 *  - Read voltage on rails
 *  - Read current
 */
void modeRun() {
  if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_CHANGE_MODE)) {
    // Always enter here if you know you're going to change modes, so you don't accidently fall into the other if
    changeMode(readBuffer[1]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_RUN_POWER)) {
    runPower(readBuffer[1]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 10) && (readBuffer[0] == SERIAL_RUN_VOLTAGE)) {
    runVoltage(readBuffer[1],readBuffer+2); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_RUN_CURRENT)) {
    runCurrent(readBuffer[1]); 
    flushBuffer(); 
  }
}

void runPower(int option) {
  switch (option) {
    case SWITCH_ON:
      Serial.println(F("runPower: Turning power on")); 
      Serial.println(SERIAL_VALID_REQUEST); 
      changeReadRail(PIN_RAIL_READ_LENGTH-1); 
      digitalWrite(PIN_CIRCUIT_POWER, HIGH); 
      break; 
    case SWITCH_OFF:
      Serial.println(F("runPower: Turning power off")); 
      Serial.println(SERIAL_VALID_REQUEST); 
      digitalWrite(PIN_CIRCUIT_POWER, LOW); 
      break; 
    case SWITCH_TOGGLE:
      Serial.print(F("runPower: Toggling power. New power state: ")); 
      Serial.println(!digitalRead(PIN_CIRCUIT_POWER)); 
      Serial.println(SERIAL_VALID_REQUEST); 
      digitalWrite(PIN_CIRCUIT_POWER, !digitalRead(PIN_CIRCUIT_POWER)); 
      break; 
    default:
      Serial.print(F("runPower: Invalid option: ")); 
      Serial.println(option); 
      Serial.println(SERIAL_INVALID_REQUEST); 
      break; 
  }
}

void runVoltage(int numMeasurements, int* readRailInt) {
  bool readRail[MUX_SELECT_SIZE]; 
  for (int i = 0; i < MUX_SELECT_SIZE/8; i++) {
    for (int ii = 0; ii < 8; ii++) {
      readRail[i*8+ii] = readRailInt[i] & (1 << ii); 
    }
  }
  Serial.println(F("runVoltage: Running with these samples per rail: ")); 
  Serial.println(numMeasurements); 
  Serial.println(F("Sampling these rails:")); 
  for (int i = 0; i < MUX_SELECT_SIZE; i++) {
    if (readRail[i]) {
      Serial.print(i); 
      Serial.print(F(" ")); 
    }
  }
  Serial.println(); 
  Serial.println(SERIAL_VALID_REQUEST); 
  int railVoltage; 
  bool contentMeasurement; 
  int switchCount; 
  // TODO: Add warning if there is no power? 
  changeReadRail(PIN_RAIL_READ_LENGTH-1); 
  digitalWrite(PIN_RAIL_ENABLE, HIGH); 
  // MUX REWIRING
  //for (int i = 0; i < PIN_MUX_SELECT_LENGTH; i++) {
  //  digitalWrite(PIN_MUX_SELECT_START+i, LOW); 
  //}
  setMuxSelectMuxRewired(0); 
  digitalWrite(PIN_MUX_ENABLE, HIGH); 
  delayMicroseconds(1); 
  //delay(10000); 
  for (int i = 0; i < MUX_SELECT_SIZE; i++) {
    if (!readRail[i]) {
      continue; 
    }
    // MUX REWIRING
    //setMuxSelect(PIN_MUX_SELECT_START, PIN_MUX_SELECT_LENGTH, i); 
    setMuxSelectMuxRewired(i); 
    if (!robustRead) {
      changeReadRail(PIN_RAIL_READ_LENGTH-1); 
      delayMicroseconds(1); 
      contentMeasurement = false; 
      switchCount = 0; 
      railVoltage = 0; 
      while (!contentMeasurement && switchCount < 10) {
        railVoltage = analogRead(PIN_RAIL_READ_START+readRailSelected); 
        if ( (railVoltage >= ADC_SIZE*0.95) && (readRailSelected < (PIN_RAIL_READ_LENGTH-1)) ) {
          changeReadRail(readRailSelected+1); 
        } else if ( (float(railVoltage)/ADC_SIZE <= float(readRailSelected)/(readRailSelected+1)*0.95) && (readRailSelected > 0) ) { // TODO: Make comparison explicit to some common variable
          changeReadRail(readRailSelected-1); 
        } else {
          contentMeasurement = true; 
        }
        switchCount++; 
      }
    }
    //delay(2000); 
    railVoltage = 0; 
    for (int ii = 0; ii < numMeasurements; ii++) {
      // railVoltage[i] += analogRead(PIN_RAIL_ANALOG_READ); 
      railVoltage += analogRead(PIN_RAIL_READ_START+readRailSelected); 
      delayMicroseconds(1); 
    }
    railVoltage = railVoltage/numMeasurements; 
    Serial.println(i); 
    Serial.println(readRailSelected); 
    Serial.println(railVoltage); 
  }
  digitalWrite(PIN_RAIL_ENABLE, LOW); 
  digitalWrite(PIN_MUX_ENABLE, LOW); 
  // MUX REWIRING
  //for (int i = 0; i < PIN_MUX_SELECT_LENGTH; i++) {
  //  digitalWrite(PIN_MUX_SELECT_START+i, LOW); 
  //}
  setMuxSelectMuxRewired(0); 
}

void runCurrent(int numMeasurements) {
  // Placeholder. Will require tinkering once we get the differential ADC in place
}



/* Available commands: 
 *  - Toggle power
 *  - Toggle demux enable
 *  - Toggle demux select
 *  - Toggle mux enable
 *  - Toggle mux select
 *  - Toggle reference resistor mux enable
 *  - Toggle reference resistor mux select
 *  - Shut down all enables
 *  - Get rail analog read
 *  - Get reference resistor analog read
 *  - Toggle a pin
 *  - Get the direction of any pin
 *  - Get the state of any pin
 */
void modeDebug() {
  if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_CHANGE_MODE)) {
    // Always enter here if you know you're going to change modes, so you don't accidently fall into the other if
    changeMode(readBuffer[1]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_DEBUG_POWER)) {
    debugPower(readBuffer[1]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_DEBUG_DEMUX_ENABLE)) {
    debugDemuxEnable(readBuffer[1]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_DEBUG_DEMUX_SELECT)) {
    debugDemuxSelect(readBuffer[1]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_DEBUG_MUX_ENABLE)) {
    debugMuxEnable(readBuffer[1]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_DEBUG_MUX_SELECT)) {
    debugMuxSelect(readBuffer[1]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_DEBUG_REFERENCE_RESISTOR_ENABLE)) {
    debugReferenceResistorEnable(readBuffer[1]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_DEBUG_REFERENCE_RESISTOR_SELECT)) {
    debugReferenceResistorSelect(readBuffer[1]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 1) && (readBuffer[0] == SERIAL_DEBUG_ENABLE_SHUTDOWN)) {
    debugEnableShutdown(); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 1) && (readBuffer[0] == SERIAL_DEBUG_READ_RAIL)) {
    debugReadRail(); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 1) && (readBuffer[0] == SERIAL_DEBUG_READ_CURRENT)) {
    debugReadCurrent(); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_DEBUG_READ_REFERENCE_RESISTOR)) {
    debugReadReferenceResistor(readBuffer[1]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_DEBUG_RAIL_ENABLE)) {
    debugRailEnable(readBuffer[1]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_DEBUG_RAIL_SELECT)) {
    debugRailSelect(readBuffer[1]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_DEBUG_MUX_READ_RAIL)) {
    debugMuxReadRail(readBuffer[1]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 3) && (readBuffer[0] == SERIAL_DEBUG_SET_PIN)) {
    debugSetPin(readBuffer[1], readBuffer[2]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_DEBUG_GET_PIN_DIRECTION)) {
    debugGetPinDirection(readBuffer[1]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_DEBUG_GET_PIN_STATE)) {
    debugGetPinState(readBuffer[1]); 
    flushBuffer(); 
  //} else if ((readBufferIndex >= 4) && (readBuffer[0] == SERIAL_DEBUG_LIN_POT_WRITE)) {
  //  unsigned int data = readBuffer[2]<<8 + readBuffer[3]; 
  //  linPotWrite(readBuffer[1], data); 
  //  flushBuffer(); 
  } else if ((readBufferIndex >= 3) && (readBuffer[0] == SERIAL_DEBUG_LIN_POT_READ)) {
    linPotRead(readBuffer[1], readBuffer[2]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_DEBUG_LIN_POT_INCREMENT)) {
    linPotIncrement(readBuffer[1]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_DEBUG_LIN_POT_DECREMENT)) {
    linPotDecrement(readBuffer[1]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 3) && (readBuffer[0] == SERIAL_DEBUG_LIN_POT_WRITE_2)) {
    //unsigned int data = readBuffer[2]*(1<<8) + readBuffer[3]; 
    linPotWrite2(readBuffer[1], readBuffer[2]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 3) && (readBuffer[0] == SERIAL_DEBUG_LOG_POT_WRITE)) {
    logPotWrite(readBuffer[1], readBuffer[2]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 3) && (readBuffer[0] == SERIAL_DEBUG_LIN_POT_RR_WRITE)) {
    //unsigned int data = readBuffer[2]*(1<<8) + readBuffer[3]; 
    linPotWriteRR(readBuffer[1], readBuffer[2]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 3) && (readBuffer[0] == SERIAL_DEBUG_LIN_POT_RR_READ)) {
    linPotReadRR(readBuffer[1], readBuffer[2]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_DEBUG_LIN_POT_RR_INCREMENT)) {
    linPotIncrementRR(readBuffer[1]); 
    flushBuffer(); 
  } else if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_DEBUG_LIN_POT_RR_DECREMENT)) {
    linPotDecrementRR(readBuffer[1]); 
    flushBuffer(); 
  } 
}

void debugPower(int option) {
  switch (option) {
    case SWITCH_ON:
      Serial.println(F("debugPower: Turning power on")); 
      Serial.println(SERIAL_VALID_REQUEST); 
      digitalWrite(PIN_CIRCUIT_POWER, HIGH); 
      break; 
    case SWITCH_OFF:
      Serial.println(F("debugPower: Turning power off")); 
      Serial.println(SERIAL_VALID_REQUEST); 
      digitalWrite(PIN_CIRCUIT_POWER, LOW); 
      break; 
    case SWITCH_TOGGLE:
      Serial.print(F("debugPower: Toggling power. New power state: ")); 
      Serial.println(!digitalRead(PIN_CIRCUIT_POWER)); 
      Serial.println(SERIAL_VALID_REQUEST); 
      digitalWrite(PIN_CIRCUIT_POWER, !digitalRead(PIN_CIRCUIT_POWER)); 
      break; 
    default:
      Serial.print("debugPower: Invalid option: "); 
      Serial.println(option); 
      Serial.println(SERIAL_INVALID_REQUEST); 
      break; 
  }
}

void debugDemuxEnable(int option) {
  switch (option) {
    case SWITCH_ON:
      Serial.println(F("debugDemuxEnable: Turning demux on")); 
      Serial.println(SERIAL_VALID_REQUEST); 
      digitalWrite(PIN_DEMUX_ENABLE, HIGH); 
      break; 
    case SWITCH_OFF:
      Serial.println(F("debugDemuxEnable: Turning demux off")); 
      Serial.println(SERIAL_VALID_REQUEST); 
      digitalWrite(PIN_DEMUX_ENABLE, LOW); 
      break; 
    case SWITCH_TOGGLE:
      Serial.print(F("debugDemuxEnable: Toggling demux. New demux enable state: ")); 
      Serial.println(!digitalRead(PIN_DEMUX_ENABLE)); 
      Serial.println(SERIAL_VALID_REQUEST); 
      digitalWrite(PIN_DEMUX_ENABLE, !digitalRead(PIN_DEMUX_ENABLE)); 
      break; 
    default:
      Serial.print(F("debugDemuxEnable: Invalid option: ")); 
      Serial.println(option); 
      Serial.println(SERIAL_INVALID_REQUEST); 
      break; 
  }
}

void debugDemuxSelect(int rail) {
  if (rail >= 0 && rail < MUX_SELECT_SIZE) {
    Serial.print(F("debugDemuxSelect: Selecting rail: ")); 
    Serial.println(rail); 
    Serial.println(SERIAL_VALID_REQUEST); 
    for (int i = 0; i < PIN_DEMUX_SELECT_LENGTH; i++) {
      digitalWrite(PIN_DEMUX_SELECT_START+i, (1 << i) & rail); 
    }
  } else {
    Serial.print(F("debugDemuxSelect: Invalid rail: ")); 
    Serial.println(rail); 
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

void debugMuxEnable(int option) {
  switch (option) {
    case SWITCH_ON:
      Serial.println(F("debugMuxEnable: Turning mux on")); 
      Serial.println(SERIAL_VALID_REQUEST); 
      digitalWrite(PIN_MUX_ENABLE, HIGH); 
      break; 
    case SWITCH_OFF:
      Serial.println(F("debugMuxEnable: Turning mux off")); 
      Serial.println(SERIAL_VALID_REQUEST); 
      digitalWrite(PIN_MUX_ENABLE, LOW); 
      break; 
    case SWITCH_TOGGLE:
      Serial.print(F("debugMuxEnable: Toggling mux. New mux enable state: ")); 
      Serial.println(!digitalRead(PIN_MUX_ENABLE)); 
      Serial.println(SERIAL_VALID_REQUEST); 
      digitalWrite(PIN_MUX_ENABLE, !digitalRead(PIN_MUX_ENABLE)); 
      break; 
    default:
      Serial.print(F("debugMuxEnable: Invalid option: ")); 
      Serial.println(option); 
      Serial.println(SERIAL_INVALID_REQUEST); 
      break; 
  }
}

void debugMuxSelect(int rail) {
  if (rail >= 0 && rail < MUX_SELECT_SIZE) {
    Serial.print(F("debugMuxSelect: Selecting rail: ")); 
    Serial.println(rail); 
    Serial.println(SERIAL_VALID_REQUEST); 
    // MUX REWIRING
    //for (int i = 0; i < PIN_MUX_SELECT_LENGTH; i++) {
    //  digitalWrite(PIN_MUX_SELECT_START+i, (1 << i) & rail); 
    //}
    setMuxSelectMuxRewired(rail); 
  } else {
    Serial.print(F("debugMuxSelect: Invalid rail: ")); 
    Serial.println(rail); 
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

void debugReferenceResistorEnable(int option) {
  switch (option) {
    case SWITCH_ON:
      Serial.println(F("debugReferenceResistorEnable: Turning reference resistor on")); 
      Serial.println(SERIAL_VALID_REQUEST); 
      digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, HIGH); 
      break; 
    case SWITCH_OFF:
      Serial.println(F("debugReferenceResistorEnable: Turning reference resistor off")); 
      Serial.println(SERIAL_VALID_REQUEST); 
      digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, LOW); 
      break; 
    case SWITCH_TOGGLE:
      Serial.print(F("debugReferenceResistorEnable: Toggling reference resistor. New reference resistor enable state: ")); 
      Serial.println(!digitalRead(PIN_REFERENCE_RESISTOR_ENABLE)); 
      Serial.println(SERIAL_VALID_REQUEST); 
      digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, !digitalRead(PIN_REFERENCE_RESISTOR_ENABLE)); 
      break; 
    default:
      Serial.print(F("debugReferenceResistorEnable: Invalid option: ")); 
      Serial.println(option); 
      Serial.println(SERIAL_INVALID_REQUEST); 
      break; 
  }
}

void debugReferenceResistorSelect(int rail) {
  if ( (rail >= 0 && rail < MUX_SELECT_SIZE) || (rail == REFERENCE_RESISTOR_DRAIN_INDEX) ) {
    Serial.print(F("debugReferenceResistorSelect: Selecting rail: ")); 
    Serial.println(rail); 
    Serial.println(SERIAL_VALID_REQUEST); 
    for (int i = 0; i < PIN_REFERENCE_RESISTOR_SELECT_LENGTH; i++) {
      digitalWrite(PIN_REFERENCE_RESISTOR_SELECT_START+i, (1 << i) & rail); 
    }
  } else {
    Serial.print(F("debugReferenceResistorSelect: Invalid rail: ")); 
    Serial.println(rail); 
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

void debugEnableShutdown() {
  Serial.println(F("debugEnableShutdown: Disabling all rails on switches and muxes")); 
  Serial.println(SERIAL_VALID_REQUEST); 
  digitalWrite(PIN_CIRCUIT_POWER, LOW); 
  digitalWrite(PIN_DEMUX_ENABLE, LOW); 
  digitalWrite(PIN_MUX_ENABLE, LOW); 
  digitalWrite(PIN_REFERENCE_RESISTOR_ENABLE, LOW); 
}

void debugReadRail() {
  Serial.print(F("debugReadRail: Reading rail analog voltage")); 
  Serial.println(SERIAL_VALID_REQUEST); 
  Serial.println(analogRead(PIN_RAIL_ANALOG_READ)); 
}

void debugReadCurrent() {
  // Placeholder for current power stuff
}

void debugReadReferenceResistor(int select) {
  if (select >= 0 && select < REFERENCE_RESISTOR_SELECT_SIZE) {
    Serial.print(F("debugReadReferenceResistor: Reading select: ")); 
    Serial.println(select); 
    Serial.println(SERIAL_VALID_REQUEST); 
    Serial.println(analogRead(PIN_REFERENCE_RESISTOR_ANALOG_READ_START+select)); 
  } else {
    Serial.print(F("debugReadRail: Invalid select: ")); 
    Serial.println(select); 
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

void debugRailEnable(int option) {
  switch (option) {
    case SWITCH_ON:
      Serial.println(F("debugRailEnable: Turning rail enable on")); 
      Serial.println(SERIAL_VALID_REQUEST); 
      digitalWrite(PIN_RAIL_ENABLE, HIGH); 
      break; 
    case SWITCH_OFF:
      Serial.println(F("debugRailEnable: Turning rail enable off")); 
      Serial.println(SERIAL_VALID_REQUEST); 
      digitalWrite(PIN_RAIL_ENABLE, LOW); 
      break; 
    case SWITCH_TOGGLE:
      Serial.print(F("debugRailEnable: Toggling railenable. New rail enable state: ")); 
      Serial.println(!digitalRead(PIN_RAIL_ENABLE)); 
      Serial.println(SERIAL_VALID_REQUEST); 
      digitalWrite(PIN_RAIL_ENABLE, !digitalRead(PIN_RAIL_ENABLE)); 
      break; 
    default:
      Serial.print(F("debugRailEnable: Invalid option: ")); 
      Serial.println(option); 
      Serial.println(SERIAL_INVALID_REQUEST); 
      break; 
  }
}
  
void debugRailSelect(int rail) {
  if (rail >= 0 && rail < PIN_RAIL_READ_LENGTH) {
    Serial.print(F("debugRailSelect: Selecting rail: ")); 
    Serial.println(rail); 
    Serial.println(SERIAL_VALID_REQUEST); 
    for (int i = 0; i < PIN_RAIL_SELECT_LENGTH; i++) {
      digitalWrite(PIN_RAIL_SELECT_START+i, (1 << i) & rail); 
    }
  } else {
    Serial.print(F("debugRailSelect: Invalid rail: ")); 
    Serial.println(rail); 
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

void debugMuxReadRail(int rail) {
  if (rail >= 0 && rail < PIN_RAIL_READ_LENGTH) {
    Serial.print(F("debugMuxReadRail: Reading rail: ")); 
    Serial.println(rail); 
    Serial.println(SERIAL_VALID_REQUEST); 
    Serial.println(analogRead(PIN_RAIL_READ_START+readRailSelected)); 
  } else {
    Serial.print(F("debugMuxReadRail: Invalid select: ")); 
    Serial.println(rail); 
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

void debugSetPin(int pin, int option) {
  if (pin >= 0 && pin < NUM_PINS) {
    if (getPinMode(pin) == OUTPUT) {
      Serial.print(F("debugSetPin: Valid pin: ")); 
      Serial.println(pin); 
      switch (option) {
        case SWITCH_ON:
          Serial.println(F("Turning demux on")); 
          Serial.println(SERIAL_VALID_REQUEST); 
          digitalWrite(PIN_DEMUX_ENABLE, HIGH); 
          break; 
        case SWITCH_OFF:
          Serial.println(F("Turning demux off")); 
          Serial.println(SERIAL_VALID_REQUEST); 
          digitalWrite(PIN_DEMUX_ENABLE, LOW); 
          break; 
        case SWITCH_TOGGLE:
          Serial.print(F("Toggling demux. New demux enable state: ")); 
          Serial.println(!digitalRead(PIN_DEMUX_ENABLE)); 
          Serial.println(SERIAL_VALID_REQUEST); 
          digitalWrite(PIN_DEMUX_ENABLE, !digitalRead(PIN_DEMUX_ENABLE)); 
          break; 
        default:
          Serial.print(F("Invalid option: ")); 
          Serial.println(option); 
          Serial.println(SERIAL_INVALID_REQUEST); 
          break; 
      }
    } else {
      Serial.print(F("debugSetPin: Pin is not an output")); 
      Serial.println(SERIAL_INVALID_REQUEST); 
    }
  } else {
    Serial.print(F("debugSetPin: Invalid pin: ")); 
    Serial.println(pin); 
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

void debugGetPinDirection(int pin) {
  if (pin >= 0 && pin < NUM_PINS) {
    Serial.print(F("debugGetPinDirection: Valid pin: ")); 
    Serial.println(pin); 
    Serial.println(SERIAL_VALID_REQUEST); 
    Serial.println(getPinMode(pin)); 
  } else {
    Serial.print(F("debugGetPinDirection: Invalid pin: ")); 
    Serial.println(pin); 
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

void debugGetPinState(int pin) {
  if (pin >= 0 && pin < NUM_PINS) {
    Serial.print(F("debugGetPinState: Valid pin: ")); 
    Serial.println(pin); 
    Serial.println(SERIAL_VALID_REQUEST); 
    Serial.println(digitalRead(pin)); 
  } else {
    Serial.print(F("debugGetPinState: Invalid pin: ")); 
    Serial.println(pin); 
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}




void linPotDecrement(unsigned int potNum) {
  if (potNum < 2) {
    Serial.print(F("linPotDecrement: Valid write command. PotNum: ")); 
    Serial.println(potNum); 
    Serial.println(SERIAL_VALID_REQUEST); 
    unsigned int bitstream = 0; 
    bitstream = (bitstream<<4) + potNum; 
    bitstream = (bitstream<<2) + LIN_POT_DECREMENT_COMMAND; 
    bitstream = (bitstream<<2); 
    digitalWrite(PIN_LIN_POT_CS_BAR, LOW); 
    //delay(1000); 
    SPI.transfer(bitstream); 
    digitalWrite(PIN_LIN_POT_CS_BAR, HIGH); 
  } else {
    if (potNum >= 2) {
      Serial.print(F("linPotDecrement: Invalid pot number: ")); 
      Serial.println(potNum); 
    }
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

void linPotIncrement(unsigned int potNum) {
  if (potNum < 2) {
    Serial.print(F("linPotIncrement: Valid write command. PotNum: ")); 
    Serial.println(potNum); 
    Serial.println(SERIAL_VALID_REQUEST); 
    unsigned int bitstream = 0; 
    bitstream = (bitstream<<4) + potNum; 
    bitstream = (bitstream<<2) + LIN_POT_INCREMENT_COMMAND; 
    bitstream = (bitstream<<2); 
    digitalWrite(PIN_LIN_POT_CS_BAR, LOW); 
    //delay(1000); 
    SPI.transfer(bitstream); 
    digitalWrite(PIN_LIN_POT_CS_BAR, HIGH); 
  } else {
    if (potNum >= 2) {
      Serial.print(F("linPotIncrement: Invalid pot number: ")); 
      Serial.println(potNum); 
    }
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

void linPotRead(unsigned int potNum, unsigned int data) {
  if (potNum < 2 && data < 257) {
    Serial.print(F("linPotRead: Valid read command. PotNum: ")); 
    Serial.print(potNum); 
    Serial.print(F(", Data: ")); 
    Serial.println(data); 
    Serial.println(SERIAL_VALID_REQUEST); 
    unsigned int bitstream = 0; 
    bitstream = (bitstream<<4) + potNum; 
    bitstream = (bitstream<<2) + LIN_POT_READ_COMMAND; 
    bitstream = (bitstream<<10) + data; 
    digitalWrite(PIN_LIN_POT_CS_BAR, LOW); 
    //delay(1000); 
    SPI.transfer16(bitstream); 
    digitalWrite(PIN_LIN_POT_CS_BAR, HIGH); 
    // Code to read bitstream
  } else {
    if (potNum >= 2) {
      Serial.print(F("linPotRead: Invalid pot number: ")); 
      Serial.println(potNum); 
    }
    if (data >= 256) {
      Serial.print(F("linPotRead: Invalid data stream: ")); 
      Serial.println(data); 
    }
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

/*
void linPotWrite(unsigned int potNum, unsigned int data) {
  if (potNum < 2 && data < (1<<10)) {
    Serial.print(F("linPotWrite: Valid write command. PotNum: ")); 
    Serial.print(potNum); 
    Serial.print(F(", Data: ")); 
    Serial.println(data); 
    Serial.println(SERIAL_VALID_REQUEST); 
    unsigned int bitstream = 0; 
    bitstream = (bitstream<<4) + potNum; 
    bitstream = (bitstream<<2) + LIN_POT_WRITE_COMMAND; 
    bitstream = (bitstream<<2) + 1; //Enable wiper
    bitstream = (bitstream<<8) + data; 
    linPotSerial(bitstream); 
  } else {
    if (potNum >= 2) {
      Serial.print(F("linPotWrite: Invalid pot number: ")); 
      Serial.println(potNum); 
    }
    if (data >= (1<<10)) {
      Serial.print(F("linPotWrite: Invalid data stream: ")); 
      Serial.println(data); 
    }
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}
*/

/*
void linPotSerial(unsigned int bitstream) {
  digitalWrite(PIN_LIN_POT_SCK, LOW); 
  digitalWrite(PIN_LIN_POT_CS_BAR, LOW); 
  int commandLength = 0; 
  if (bitstream < 256) {
    commandLength = 8; 
  } else {
    commandLength = 16; 
  }
  for (int i = commandLength-1; i >= 0; i--) {
    digitalWrite(PIN_LIN_POT_SDI, (1 << i) & bitstream); 
    digitalWrite(PIN_LIN_POT_SCK, HIGH); 
    digitalWrite(PIN_LIN_POT_SCK, LOW); 
  }
  digitalWrite(PIN_LIN_POT_CS_BAR, HIGH); 
}
*/

void linPotWrite2(unsigned int potNum, unsigned int data) {
  if (potNum < 2 && data < 257) {
    Serial.print(F("linPotWrite2: Valid write command. PotNum: ")); 
    Serial.print(potNum); 
    Serial.print(F(", Data: ")); 
    Serial.println(data); 
    Serial.println(SERIAL_VALID_REQUEST); 
    unsigned int bitstream = 0; 
    bitstream = (bitstream<<4) + potNum; 
    bitstream = (bitstream<<2) + LIN_POT_WRITE_COMMAND; 
    bitstream = bitstream<<2; 
    //bitstream = (bitstream<<2) + 1; //Enable wiper
    bitstream = (bitstream<<8) + data; 
    //bitstream = (bitstream<<10) + data; 
    digitalWrite(PIN_LIN_POT_CS_BAR, LOW); 
    //delay(1000); 
    SPI.transfer16(bitstream); 
    digitalWrite(PIN_LIN_POT_CS_BAR, HIGH); 
    //delay(1000); 
  } else {
    if (potNum >= 2) {
      Serial.print(F("linPotWrite2: Invalid pot number: ")); 
      Serial.println(potNum); 
    }
    if (data >= 257) {
      Serial.print(F("linPotWrite2: Invalid data stream: ")); 
      Serial.println(data); 
    }
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

void linPotWriteRR(unsigned int potNum, unsigned int data) {
  if (potNum < 2 && data < 257) {
    Serial.print(F("linPotWriteRR: Valid write command. PotNum: ")); 
    Serial.print(potNum); 
    Serial.print(F(", Data: ")); 
    Serial.println(data); 
    Serial.println(SERIAL_VALID_REQUEST); 
    unsigned int bitstream = 0; 
    bitstream = (bitstream<<4) + potNum; 
    bitstream = (bitstream<<2) + LIN_POT_WRITE_COMMAND; 
    bitstream = bitstream<<2; 
    //bitstream = (bitstream<<2) + 1; //Enable wiper
    bitstream = (bitstream<<8) + data; 
    //bitstream = (bitstream<<10) + data; 
    digitalWrite(PIN_LIN_POT_RR_CS_BAR, LOW); 
    //delay(1000); 
    SPI.transfer16(bitstream); 
    digitalWrite(PIN_LIN_POT_RR_CS_BAR, HIGH); 
    //delay(1000); 
  } else {
    if (potNum >= 2) {
      Serial.print(F("linPotWriteRR: Invalid pot number: ")); 
      Serial.println(potNum); 
    }
    if (data >= 257) {
      Serial.print(F("linPotWriteRR: Invalid data stream: ")); 
      Serial.println(data); 
    }
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}


void linPotDecrementRR(unsigned int potNum) {
  if (potNum < 2) {
    Serial.print(F("linPotDecrement: Valid write command. PotNum: ")); 
    Serial.println(potNum); 
    Serial.println(SERIAL_VALID_REQUEST); 
    unsigned int bitstream = 0; 
    bitstream = (bitstream<<4) + potNum; 
    bitstream = (bitstream<<2) + LIN_POT_DECREMENT_COMMAND; 
    bitstream = (bitstream<<2); 
    digitalWrite(PIN_LIN_POT_RR_CS_BAR, LOW); 
    //delay(1000); 
    SPI.transfer(bitstream); 
    digitalWrite(PIN_LIN_POT_RR_CS_BAR, HIGH); 
  } else {
    if (potNum >= 2) {
      Serial.print(F("linPotDecrement: Invalid pot number: ")); 
      Serial.println(potNum); 
    }
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

void linPotIncrementRR(unsigned int potNum) {
  if (potNum < 2) {
    Serial.print(F("linPotIncrement: Valid write command. PotNum: ")); 
    Serial.println(potNum); 
    Serial.println(SERIAL_VALID_REQUEST); 
    unsigned int bitstream = 0; 
    bitstream = (bitstream<<4) + potNum; 
    bitstream = (bitstream<<2) + LIN_POT_INCREMENT_COMMAND; 
    bitstream = (bitstream<<2); 
    digitalWrite(PIN_LIN_POT_RR_CS_BAR, LOW); 
    //delay(1000); 
    SPI.transfer(bitstream); 
    digitalWrite(PIN_LIN_POT_RR_CS_BAR, HIGH); 
  } else {
    if (potNum >= 2) {
      Serial.print(F("linPotIncrement: Invalid pot number: ")); 
      Serial.println(potNum); 
    }
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

void linPotReadRR(unsigned int potNum, unsigned int data) {
  if (potNum < 2 && data < 257) {
    Serial.print(F("linPotRead: Valid read command. PotNum: ")); 
    Serial.print(potNum); 
    Serial.print(F(", Data: ")); 
    Serial.println(data); 
    Serial.println(SERIAL_VALID_REQUEST); 
    unsigned int bitstream = 0; 
    bitstream = (bitstream<<4) + potNum; 
    bitstream = (bitstream<<2) + LIN_POT_READ_COMMAND; 
    bitstream = (bitstream<<10) + data; 
    digitalWrite(PIN_LIN_POT_RR_CS_BAR, LOW); 
    //delay(1000); 
    SPI.transfer16(bitstream); 
    digitalWrite(PIN_LIN_POT_RR_CS_BAR, HIGH); 
    // Code to read bitstream
  } else {
    if (potNum >= 2) {
      Serial.print(F("linPotRead: Invalid pot number: ")); 
      Serial.println(potNum); 
    }
    if (data >= 256) {
      Serial.print(F("linPotRead: Invalid data stream: ")); 
      Serial.println(data); 
    }
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

void logPotWrite(unsigned int potNum, unsigned int data) {
  static unsigned int pot0 = 0; 
  static unsigned int pot1 = 0; 
  if (potNum < 2 && data < 65) {
    //Serial.print(F("Reverse bitstream: ")); 
    //Serial.println(reverseBitOrder(bitstream)); 
    Serial.print(F("logPotWrite: Valid write command. PotNum: ")); 
    Serial.print(potNum); 
    Serial.print(F(", Data: ")); 
    Serial.println(data); 
    Serial.print(F("Pot 0: ")); 
    Serial.print(pot0); 
    Serial.print(F(", Pot 1: ")); 
    Serial.println(pot1); 
    Serial.println(SERIAL_VALID_REQUEST);
    if (potNum == 0) {
      pot0 = data; 
    } else if (potNum == 1) {
      pot1 = data; 
    }
    unsigned int bitstream = 0; 
    bitstream = (bitstream<<1); 
    bitstream = (bitstream<<7) + pot1; 
    bitstream = (bitstream<<1); 
    bitstream = (bitstream<<7) + pot0; 
    // LOG Enable is enabled when high
    digitalWrite(PIN_LOG_POT_ENABLE, HIGH); 
    //delay(1000); 
    // LOG pot is shifted in LSB first
    //SPI.transfer16(bitstream); 
    //delay(10000); 
    //SPI.transfer16(reverseBitOrder(bitstream)); 
    spiSerial(bitstream); 
    //spiSerial(-1); 
    //delay(1000); 
    //delay(1000); 
    //spiSerial(-1); 
    //SPI.transfer16(0); 
    //delay(1000); 
    //SPI.transfer16(-1); 
    //delay(1000); 
    //delay(30000); 
    digitalWrite(PIN_LOG_POT_ENABLE, LOW);  
  } else {
    if (potNum >= 2) {
      Serial.print(F("logPotWrite: Invalid pot number: ")); 
      Serial.println(potNum); 
    }
    if (data >= 65) {
      Serial.print(F("logPotWrite: Invalid data stream: ")); 
      Serial.println(data); 
    }
    Serial.println(SERIAL_INVALID_REQUEST); 
  }
}

void spiSerial(unsigned int bitstream) {
  digitalWrite(PIN_SPI_SCK, LOW); 
  int commandLength = 16; 
  //if (bitstream < 256) {
  //  commandLength = 8; 
  //} else {
  //  commandLength = 16; 
  //}
  //for (int i = commandLength-1; i >= 0; i--) {
  for (int i = 0; i < 16; i++) {
    digitalWrite(PIN_SPI_SDI, ((1 << i) & bitstream) != 0); 
    digitalWrite(PIN_SPI_SCK, HIGH); 
    //delay(500); 
    digitalWrite(PIN_SPI_SCK, LOW); 
    //delay(500); 
  }
}



void modeUnknown() {
  if ((readBufferIndex >= 2) && (readBuffer[0] == SERIAL_CHANGE_MODE)) {
    // Always enter here if you know you're going to change modes, so you don't accidently fall into the other if
    changeMode(readBuffer[1]); 
    flushBuffer(); 
  }
}

unsigned int reverseBitOrder(unsigned int vector) {
  unsigned int reversedVector = 0; 
  for (int i = 0; i < sizeof(unsigned int); i++) {
    reversedVector += (vector == (1<<i)) * (1<<(sizeof(unsigned int)-i)); 
  }
  return reversedVector; 
}

void setMuxSelect(int pinStart, int pinLength, int value) {
  for (int i = 0; i < pinLength; i++) {
    digitalWrite(pinStart+i, (1 << i) & value); 
  }
}

// This special routine exists because I remapped the wiring of the mux to better fit on to the pcb
// All that's changed, circuit-wise is that the order for each 2nd layer mux now goes from
// 0, 1, 2, 3, 4, 5, 6, 7 -> 4, 5, 6, 7, 0, 1, 2, 3
// Which means that to reference each mux correctly given the initial value you need to invert the highest order bit
void setMuxSelectMuxRewired(int value) {
  for (int i = 0; i < PIN_MUX_SELECT_LENGTH; i++) {
    digitalWrite(PIN_MUX_SELECT_START+i, (1 << i) & (value ^ (1<<2))); 
    //digitalWrite(PIN_MUX_SELECT_START+i, (1 << i) & value); 
  }
}

int getPinMode(int pin) {
  if (pin >= NUM_PINS) return (-1);

  uint8_t bitPin = digitalPinToBitMask(pin);
  uint8_t port = digitalPinToPort(pin);
  volatile uint8_t *reg = portModeRegister(port);
  if (*reg & bitPin) return (OUTPUT);

  volatile uint8_t *out = portOutputRegister(port);
  return ((*out & bitPin) ? INPUT_PULLUP : INPUT);
}
