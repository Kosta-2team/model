int buzzer = 2;
int echo = 12;
int trig = 13; 
float duration;
float distance;
int thresholdDistance = 15;

bool vehicleDetected = false;  // 차량이 범위 내에 있는 상태

void setup() 
{
  pinMode(buzzer, OUTPUT);
  pinMode(trig, OUTPUT);
  pinMode(echo, INPUT);
  Serial.begin(9600);
}

void loop() 
{
  // 초음파 신호 보내기
  digitalWrite(trig, LOW);
  delayMicroseconds(2);
  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);

  // 초음파 수신 시간 측정
  duration = pulseIn(echo, HIGH);
  distance = (duration * 0.034) / 2; 

  // 거리 출력
  // Serial.print("Distance: ");
  // Serial.println(distance);

    // 차량 진입 감지
  if (distance <= thresholdDistance && !vehicleDetected) {
    vehicleDetected = true;         
    Serial.println("ENTRY"); // 입차시 신호 전송       
    //playEntrySound();               
  }

  delay(1000); 

  vehicleDetected = false;         

}

// 차량 진입 소리 함수 (긴 소리)
void playEntrySound() {
  digitalWrite(buzzer, HIGH);
  delay(1000); 
  digitalWrite(buzzer, LOW);
}
