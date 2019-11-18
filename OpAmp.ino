/*
  OpAmp: Sketch to measure the temperature with a thermistor and the grove temperature sensor
*/
#include <math.h>
#include <Wire.h>
#include "rgb_lcd.h"

rgb_lcd lcd;

short int vol; // measured voltage
float r; // Resistance
float temp; // temperature from thermistor
short int degree; // measured rotation
float temp_cal; // temperature from grove sensor

#define r0 100000 // cold resistance
#define t0 25  // room temperature
#define b1 4275 // constant of the thermistor
#define b2 4600 // constant of the calibrated sensor
#define KELVIN 273.15

#define POTI A1  // pin of the potentiometer
#define FULL_ANGLE 300  // full angle of the potentiometer

#define SENSOR A2 // pin of the thermistor
#define CALIBR A0  // pin of the grove temperature sensor

void setup() {
  pinMode(POTI,INPUT);
  pinMode(SENSOR,INPUT);
  pinMode(CALIBR,INPUT);
  Serial.begin(9600);
  lcd.begin(16, 2);
}

void loop() {
  // calculation of temperature measured by thermistor
  vol = analogRead(SENSOR);
  r = (1023.0 / vol - 1.0) * r0;
  temp = log(r / r0) / b1 + 1.0 / (t0 + KELVIN);
  temp = 1.0/temp - KELVIN;

  //send the resistance from thermistor to serial
  Serial.print(millis());
  Serial.print(';');
  Serial.print(r);

  // calculate temperature of grove sensor
  vol = analogRead(CALIBR);
  r = (1023.0 / vol - 1.0) * r0;
  temp_cal = log(r / r0) / b2 + 1.0 / (t0 + KELVIN);
  temp_cal = 1.0/temp_cal - KELVIN;

  // write temp to serial
  Serial.print(';');
  Serial.println(temp_cal);
  
  // write temperature of both to lcd
  lcd.setCursor(0,0);
  lcd.print("Temp");
  lcd.print(temp);
  lcd.print(" ");
  lcd.print(temp_cal);

  delay(500);
  lcd.clear();
}
