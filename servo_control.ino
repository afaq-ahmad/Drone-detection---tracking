#include <Servo.h>

const int servo1 = 11;       // first servo
const int servo2 = 10;       // second servo
const int servo1_input = 3;        
const int servo2_input = 4;        

int servoVal;           // variable to read the value from the analog pin

Servo myservo1;  // create servo object to control a servo
Servo myservo2;  // create servo object to control a servo

void setup() {

  // Servo  
  myservo1.attach(servo1);  // attaches the servo
  myservo2.attach(servo2);  // attaches the servo

  // Inizialize Serial
  Serial.begin(9600);
}


void loop(){

    servoVal = analogRead(servo1_input);          
    servoVal = map(servoVal, 0, 1023, 0, 180);     // scale it to use it with the servo (result  between 0 and 180)

    myservo2.write(servoVal);                         // sets the servo position according to the scaled value    

    servoVal = analogRead(servo2_input);           
    servoVal = map(servoVal, 0, 1023, 70, 180);     // scale it to use it with the servo (result between 70 and 180)
    myservo1.write(servoVal);                           // sets the servo position according to the scaled value

    delay(15);                                       // waits for the servo to get there

}
