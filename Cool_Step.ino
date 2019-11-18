/*
  Cool_Step: stepwise cooling of the resistor, measuring the tempreature and fanspeed
 */
#include <math.h>

short int vol; // measured voltage
float degree; // deflection of poti
float r; // Resistance
float temp; // temperature
float rpm;  // rpm of the fan
unsigned long now;  // time of the last measurement
unsigned long now2; // time of the last speed change
unsigned short hes_count; // counted pulses of the fan speed
unsigned short period;    // period of which the pulses were counted
bool pulse = true; // HIGH = true, LOW = false
int k = 0;  // fan speed

#define r0 100000 // zero resistance
#define t0 25  // room temperature
#define b 4275 // thermistor constant

#define FAN 10  // pin fan
#define SPEED A4 // pin tacho
#define SENSOR A2  // pin thermistor

#define FREQ 500.0  // period of temp measurement
#define FREQ2 180000.0 // period
#define ANALOG_HIGH 500 // pulse identifier voltage
#define KELVIN 273.15

void setup() {
  pinMode(FAN,OUTPUT);
  pinMode(SPEED,INPUT_PULLUP);
  pinMode(SENSOR,INPUT);
  analogWrite(FAN,0);
  Serial.begin(9600);
  delay(2000);
  now = millis(); now2 = now;
}

void loop() {
  // loop for the pulse counter
  if(analogRead(SPEED)<ANALOG_HIGH && pulse) {
    hes_count++;
    pulse = false;
  }
  else  if (analogRead(SPEED)>ANALOG_HIGH && !pulse) pulse = true;
  
  // temperature measurement and rpm calculation
  if ( millis()-now > FREQ){
    // rpm calculation
    period = millis()-now;
    rpm = hes_count/2.0/FREQ*1000.0*60.0;

    //temp measurement
    vol = analogRead(SENSOR);
    r = (1023.0 / vol - 1.0) * r0;
    temp = log(r / r0) / b2 + 1.0 / (t0 + KELVIN);
    temp = 1.0/temp - KELVIN;

    // write the data to serial
    Serial.print(millis());
    Serial.print(";");
    Serial.print(temp);
    Serial.print(";");
    Serial.print(rpm);
    Serial.print(";");
    Serial.println(k);

    // reset the variables
    hes_count=0; now=millis();
  }
  
  // loop for the speed increase of the fan
  if (millis() - now2 > FREQ2){
    if (k==255 || k==0) k=15;
    else k+=40;
    analogWrite(FAN,k);
    now2 = millis();
  }
  delay(3);
}
