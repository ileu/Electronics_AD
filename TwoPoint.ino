/*
  TwoPoint: A Two Point controller for the temperature.
*/
#include <math.h>
#include <Wire.h>
#include "rgb_lcd.h"
rgb_lcd lcd;

short int vol; // measured voltage
float r; // Resistance r0
float temp; // temperature
short int degree; // measured rotation
float T_HIGH;  // upper limit of temp
float T_LOW;  // lower limit of temp

#define DELTA_T 2 // temp diff

#define r0 100000 // zero resistance
#define t0 25 // room temperature
#define b 4275 // thermistor constant

#define POTI A1 // pin potentiometer
#define FULL_ANGLE 300  // full angle 
#define SENSOR A2 // pin thermistor
#define FAN 10 // pin of pmw cont

// hot temp rgb color
#define COLOR1 255, 0, 0

// prefered temp rgb color
#define COLOR2 0, 255, 0

// cool temp rbg color 
#define COLOR3 0, 0, 255

#define TEMP_A 20 // low threshold
#define TEMP_B 40 // high threshold
#define KELVIN 273.15

void setup() {
  pinMode(POTI,INPUT);
  pinMode(FAN,OUTPUT);
  pinMode(SENSOR,INPUT);
  analogWrite(FAN,0);
  Serial.begin(9600);
  lcd.begin(16, 2);
}

void loop() {
  // measure and calculate the temperature
  vol = analogRead(SENSOR);
  r = (1023.0 / vol - 1.0) * r0;
  temp = log(r / r0) / b + 1.0 / (t0 + KELVIN);
  temp = 1.0/temp - KELVIN;

  // send data to the serial
  Serial.print(millis());
  Serial.print(';');
  Serial.println(temp);
  
  // read the potentiometer
  degree = analogRead(POTI); 
  T_HIGH = map(degree, 0, 1023, TEMP_A, TEMP_B);
  T_LOW = T_HIGH - DELTA_T;
  
  // print temperature on lcd
  lcd.setCursor(0,0);
  lcd.print("Temper");
  lcd.print(temp);
  delay(10);

  // print threshold on lcd
  lcd.setCursor(0,1);
  lcd.print("Threshold:");
  lcd.print(T_HIGH);
  delay(10);

  // color display according to temperature
  if (temp > T_HIGH){
      analogWrite(FAN,255);
      lcd.setRGB(COLOR1);
  }
  else if (temp < T_LOW){
      analogWrite(FAN,0);
      lcd.setRGB(COLOR3);
  }
  else lcd.setRGB(COLOR2);
  delay(500);
  lcd.clear();
}
