/*
 * FanSpeed: Measuring the rpm versus the pmw signal send to the fan 
 */

float degree; // measured deflection of the potentiometer
float rpm;  // calculated rpm of the fan
int vel=0;  // value of the send pmw signal to the fan
unsigned long now;  // time of last measurement
unsigned short hes_count;  // counted pulses of the fan speed
unsigned short period;  // period of which the pulses were counted
bool pulse = true; // HIGH = true, LOW = false

#define FAN 10  // pin of the fan
#define SPEED A4 // pin of the fan speed reading

#define FREQ 1000 // frequencies of the measurements
// if the pulse of the speed reading from the fan is 
// below this value we count it as a pulse
#define ANALOG_HIGH 500 

#define FULL_PMW 255  // maximum possible pmw signal

void setup() {
  pinMode(FAN,OUTPUT);
  pinMode(SPEED,INPUT_PULLUP);
  analogWrite(FAN,125);
  Serial.begin(9600);
  delay(2000);
  now = millis();
}

void loop() {

  // counting how many times the voltages drops on the pin SPEED
  if(analogRead(SPEED)<ANALOG_HIGH && pulse) {
    hes_count++;
    pulse = false;
  }
  else  if (analogRead(SPEED)>ANALOG_HIGH && !pulse) pulse = true;

  // if the time difference from the last measurement and now is bigger than FREQ
  // we calculate the rpm and send the pmw value and the rpm to the serial
  if ( millis()-now > FREQ){
    period = millis()-now;
    rpm = hes_count/2.0/period*1000.0*60;
    
    Serial.print(vel);
    Serial.print(";");
    Serial.println(rpm);
    
    //increase velocity of the fan
    vel++;
    analogWrite(FAN,vel);

    // check if vel is bigger than 255
    if(vel==FULL_PMW) vel=0;

    //reset variables
    hes_count=0;
    now=millis();
  }
}
