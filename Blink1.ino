/*
Blink1: Flashes the LED connected to pin 5 all 0.5 seconds
*/
#define LED 4

void setup() {
  pinMode(LED,OUTPUT);
}

void loop() {
  digitalWrite(LED,HIGH);
  delay(500);
  digitalWrite(LED,LOW);
  delay(500);
}
