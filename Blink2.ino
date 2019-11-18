/*
Blink2: Lets 2 LED blink with different frequencies
*/

bool d1 = 0; // if led 1 is on or off
bool d2 = 0; // if led 2 is on or off
// last time led 1 was changed
unsigned long int tick1;
// last time led 2 was changed
unsigned long int tick2;

//LED
#define LED1 4
#define LED2 5

// periods in microseconds
#define PER1  500000
#define PER2  200000

void setup() {
  pinMode(LED1,OUTPUT);
  pinMode(LED2,OUTPUT);
  tick1 = micros();
  tick2 = micros();
}

void loop() {
  if (micros() - tick1 >= PER1){
    tick1 = micros();
    if (!d1) digitalWrite(LED1, HIGH);
    else digitalWrite(LED1,LOW);
    d1=!d1;
  }
  
  if (micros() - tick2 >= PER2){
    tick2 = micros();
    if (!d2) digitalWrite(LED2, HIGH);
    else digitalWrite(LED2,LOW);
    d2=!d2;
  }
}
