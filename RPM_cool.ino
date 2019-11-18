/*
 * RPM_cool: 
 */
#include <math.h>

short int vol; // measured voltage
float r; // Resistance
float temp; // temperature
float rpm;  // calculated rpm
unsigned long now;  // time of last measurement
unsigned short hes_count;  // counted pulses of the fan speed
unsigned short period; // period of counted pulses
bool pulse = true; // HIGH = true, LOW = false

#define r0 100000 // zero resistance
#define t0 25  // room temperature
#define b 4715 // thermistor constant

#define FAN 10  // pin of the fan
#define SPEED A4 // pin of the fan speed reading
#define SENSOR A2

#define FREQ 1000.0 // period of the measurements
#define ANALOG_HIGH 500 // pulse identifier voltage

#define FULL_PMW 255
#define KELVIN 273.15

void setup() {
  pinMode(POTI,INPUT);
  pinMode(FAN,OUTPUT);
  pinMode(SPEED,INPUT_PULLUP);
  analogWrite(FAN,255);
  Serial.begin(9600);
  delay(2000);
  now = millis();
}

void loop() {
  // put your main code here, to run repeatedly:
  if(analogRead(SPEED)<ANALOG_HIGH && pulse) {
    hes_count++;
    pulse = false;
  }
  else  if (analogRead(SPEED)>ANALOG_HIGH && !pulse) pulse = true;
  
  
  if ( millis()-now > FREQ){
    // calculate the rpm of the fan
    period = millis()-now;
    rpm = hes_count/2.0/period*1000.0*60.0;
    
    // calculate the temperature
    vol = analogRead(SENSOR);
    r = (1023.0 / vol - 1.0) * r0;
    temp = log(r / r0) / b + 1.0 / (t0 + KELVIN);
    temp = 1.0/temp - KELVIN;

    // print to serial
    Serial.print(millis());
    Serial.print(";");
    Serial.print(temp);
    Serial.print(";");
    Serial.println(rpm);
    
    hes_count=0;
    now=millis();
  }
}
