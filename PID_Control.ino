/*
  sketch to measure the temperature with a Grove - Temperature sensor
*/
#include <math.h>

int rpm;  // pmw value received from the computer
unsigned long now;  // time of last measurement
unsigned short hes_count;  // counted pulses of the fan speed
bool pulse = true; // HIGH = true, LOW = false

#define SENSOR A2 // pin thermistor
#define FAN 10    // pin speed controller
#define SPEED A4  // pin speed reader

#define FREQ 200  // period of measurements
#define ANALOG_HIGH 500  // identifier for pulse

//mehtod for printing the measurements to serial
void output(unsigned long t){
    Serial.print(t);
    Serial.print(";");
    Serial.print(analogRead(SENSOR));
    Serial.print(";");
    Serial.println(hes_count);
}

void setup() {
  pinMode(FAN,OUTPUT);
  pinMode(SPEED,INPUT_PULLUP);
  pinMode(SENSOR,INPUT);
  analogWrite(FAN,0);
  Serial.begin(9600);
  delay(20);
  //ready signal for measurements
  Serial.println("ready");
  delay(30);
  // initial measurement
  output(millis());
  delay(30);
  
  while (!Serial.available()){
    // wait for instruction
  }
  // read the instruction
  rpm = Serial.read();
  analogWrite(FAN,rpm);
  delay(30);
}

void loop(){
  // pulse counter for fan speed
  if(analogRead(SPEED)<ANALOG_HIGH && pulse) {
    hes_count++;
    pulse = false;
  }
  else  if (analogRead(SPEED)>ANALOG_HIGH && !pulse) pulse = true;

  if ( millis()-now > FREQ){
    now=millis();
    //print data to serial
    output(now);
    // reset counter
    hes_count=0;
    delay(30);
    // wait for instruction and then read them and execute
    while (!Serial.available()){}
    rpm = Serial.read();
    analogWrite(FAN,rpm);
    now=millis();
  }
}
