/*
  sketch to measure the temperature with a Grove - Temperature sensor
*/
#include <math.h>
#include <Wire.h>
#include "rgb_lcd.h"
rgb_lcd lcd;

short int vol; // measured voltage
float r; // Resistance r0 * exp(-b(1/t0-1/t)
float temp; // temperature 1/t = ln(r/r0)/B + 1/t0
short int degree; // measured rotation
float threshold; // threshold of the ok temperature

#define r0 100000 // cold resistance
#define t0 25  // room temperature
#define b 4275 // constant of the temperature sensor

#define POTI A1  // pin of the potentiometer
#define FULL_ANGLE 300  // full angle of the potentiometer

#define SENSOR A0 // pin of the grove temperature sensor

// hot temp rgb color
#define COLOR1 255, 0, 0

// ok temp rgb color
#define COLOR2 0, 255, 0

#define TEMP_A 20   // lowest threshold temperature
#define TEMP_B 40   // highest threshold temperature
#define KELVIN 273.15  // from clecius to kelvin

void setup() {
  pinMode(POTI,INPUT);
  pinMode(SENSOR,INPUT);
  Serial.begin(9600);
  Serial.println("1234");
  lcd.begin(16, 2);
}

void loop() {
  // calculate the temperature from the sensor
  vol = analogRead(SENSOR);
  r = (1023.0 / vol - 1.0) * r0;
  temp = log(r / r0) / b + 1.0 / (t0 + KELVIN);
  temp = 1.0/temp - KELVIN;
  // read in the value of the potentiometer and convert to temperauter
  degree = analogRead(POTI); 
  threshold = map(degree, 0, 1023, TEMP_A, TEMP_B);

  // print on the lcd and color it accordingly
  lcd.print("Temperatur:");
  lcd.print(temp);
  if (threshold > temp) lcd.setRGB(COLOR1);
  else lcd.setRGB(COLOR2);

  lcd.setCursor(0,1);
  lcd.print("Threshold:");
  lcd.print(threshold);

  // keep for a second and then clear and redo the measurement
  delay(1000);
  lcd.clear();
}
