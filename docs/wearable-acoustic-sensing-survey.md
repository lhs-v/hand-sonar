# 약한 손가락 에코, 문헌은 배치와 주파수로 살렸다

> **바로잡음(2026-11-04):** 이 글의 "움직임 6 dB, 5–15 cm 7.7 dB, 멈춤과 정적 반사는 공개와 같다"는 칸 300 기준이다. 공개 10세션 중 6세션은 칸 300이 8 cm 뒤 둘째 도착이었다. 첫 도착에 맞추면 손 자리의 정적 반사와 움직임이 모두 우리 톱니에서 약 7–15 dB 약하다(5–15 cm 움직임 약 11 dB). 첫 손잡이를 배치로 보는 결론은 더 힘을 얻는다. 수치는 연구 현황 5.15, 그림은 보고서 hand-sonar/2026-11-03-noise-floor-knobs의 first_arrival.png에 있다.
>
> **함께 볼 것:** 저장소의 송신 신호 계산(sonar-signal.md)은 톱니 + Tukey(60)의 딸깍이 −76 dB로 삼각(−74 dB)과 같은 수준이고 한 주기 바닥은 삼각의 약 1/90이라고 냈다. 이 글의 문제 3과 함께 읽는다.

스피커·마이크로 소리를 내고 그 반사나 귀 안 변화를 읽는 웨어러블·모바일 연구는 이번에 모은 것만 80여 시스템이고, 갈래는 넷이다. Cornell SciFi Lab 계보 20편은 우리와 같은 C-FMCW 상관 에코 프로파일, 프레임 차분, 이미지 CNN으로 얼굴·무음 발화·몸·시선·먹기·인증·손을 푼다. 다른 연구실의 머리 착용 기기는 외이도 전달함수·OFDM·AR 계수나 순음 Doppler를 쓰고, 폰·스피커·손목 추적은 복소 CIR과 위상으로 mm를 만들며, 생리 연구는 긴 적분과 주파수·칸 모으기로 sub-mm 움직임을 잡는다. 기성 워치로 연속 손 자세를 낸 것은 WatchHand뿐이고, 우리 재현은 처프 모양(삼각), 녹음 경로(VOICE_RECOGNITION 대체와 튐), mm 잣대(MediaPipe 추정 손 크기) 세 곳에서만 계보를 벗어난다. 우리 실측으로는 처리로 더 얻을 몫이 작다. 줄 공통 세기·위상 빼기는 오르내림을 0.8–1.7 dB 올렸고, 포락선 차분은 오히려 1.2–1.4 dB 깎았고, 공통 회전만 뺀 위상 채널은 학습을 나쁘게 했다 (잡음 바닥 손잡이(보고서 hand-sonar/2026-11-03-noise-floor-knobs), process.json(보고서 hand-sonar/2026-11-03-noise-floor-knobs), 위상 채널(보고서 hand-sonar/2026-11-01-phase-channels)). 그래서 문제 1(움직이는 손가락 반사가 6 dB 약함)에 옮길 첫 손잡이는 문헌이 가장 크게 잰 배치다. EchoSpeech는 스피커·마이크 자리만 바꿔 정확도를 50% 미만에서 98%로 올렸고 ([EchoSpeech](https://ruidongzhang.com/files/papers/EchoSpeech_authors_version.pdf)), WatchHand 참가자는 팔꿈치를 책상에 올렸지만 우리 세션은 모두 팔 받침이 없었다. 둘째는 주파수 조각 모으기(BreathJunior·APG·스마트 스피커 심박)다. 우리 자료에서도 창 없는 17세션 모두 18–19 kHz 조각이 20–21 kHz 조각보다 손가락 변화를 더 담았다. 셋째는 칸마다 정적 벡터를 빼고 위상을 읽는 LLAP식 처리로, Galaxy Watch 6의 SonarSelect가 이 방식으로 10 cm 안에서 3–5 mm를 냈다 ([SonarSelect](https://arxiv.org/html/2503.02308v1)). 삼각 바닥(문제 3)은 상관 대신 나눗셈으로 삼각을 지킨 채 지울 수 있지만, 같은 날 견준 손 자리 오르내림 차이가 0.4 dB라 먼 칸 정리용이다. 녹음 경로(문제 4)는 Android CDD가 VOICE_RECOGNITION에 AGC·잡음 억제를 끄도록 요구하고 우리 1분 비교에서도 두 음원이 같았으니, 속성 확인과 튄 줄 버리기면 된다.

## 80여 시스템은 네 갈래이고, 기성 워치 연속 손 자세는 WatchHand 하나다

갈래는 기기와 처리로 나뉜다. 첫째는 Cornell SciFi Lab(Cheng Zhang) 계보 20편이다. 2022년 EarIO부터 2026년 WatchHand·SoniSpeech·SoniGAN까지 얼굴, 말, 몸, 시선, 먹기, 인증, 손을 모두 같은 에코 프로파일로 푼다. 연구실 목록은 2024년에서 멈춰 있어, 2025–26년 논문은 저자 검색으로 찾았다 ([SciFi Lab](https://www.scifilab.org/publications), [arXiv 저자 검색](https://arxiv.org/search/?query=Guimbreti%C3%A8re&searchtype=author)). 둘째는 다른 연구실의 머리 착용 기기다. 대부분 외이도 안의 변형이나 전달함수를 보고, 얼굴 바깥 음장이나 남의 손을 Doppler로 보는 것이 일부다. 셋째는 폰·스마트 스피커·손목의 손가락 추적과 제스처다. 넷째는 숨, 심박, 폐기능, 귀, 인증, 물질을 재는 생리·기타 연구다.

질문에 든 두 연구는 양 끝에 있다. EarIO는 개조 이어폰 양쪽에 스피커·마이크를 한 쌍씩 달고 16–20 kHz FMCW를 낸다. 착용자 얼굴에서 돌아온 에코를 거리별 프로파일의 차분으로 읽어 ARKit blendshape 52개를 회귀한다 ([EarIO](https://galaxywalk.github.io/files/imwut22eario.pdf)). SonicASL은 청자의 헤드폰에 바깥을 향한 모듈을 붙여 20 kHz 순음 하나를 낸다. 0.5–1.75 m 앞 수어 화자의 손이 만든 Doppler를 스펙트로그램으로 읽고, 거리 정보는 쓰지 않는다 ([SonicASL](https://dl.acm.org/doi/10.1145/3463519)). 손목에서 기성 기기로 돈 사례는 셋뿐이다. WatchHand는 기성 워치 세 종에서 연속 3D 손 자세를 냈고, SonarSelect는 Galaxy Watch 6에서 손등 위 손가락 하나의 1D 위치를, SonarID는 기성 워치에서 탭한 손가락을 가렸다 ([WatchHand](https://arxiv.org/html/2602.21610), [SonarSelect](https://arxiv.org/html/2503.02308v1), [SonarID](https://doi.org/10.1145/3491102.3501935)). SonarWatch는 시판 워치 마이크로는 원래 신호를 얻을 수 없다며 워치 껍데기에 외부 녹음기를 달았다 ([SonarWatch](https://arxiv.org/abs/2408.12689)).

아래 표는 응용별로 시스템을 놓는다. ID는 다음 절의 방법 표에 있다.

| 응용 | Cornell 계보 웨어러블 | 다른 웨어러블 | 폰·스피커·기타 | 대표 결과 |
|---|---|---|---|---|
| 얼굴 표정·눈 | EarIO, EyeEcho, GazeTrak(시선) | FaceListener, EARFace, ECTF, FIT2025 | BlinkListener(깜빡임) | EARFace 랜드마크 MAE 1.83 mm, GazeTrak 세션 사이 4.9° |
| 무음 발화·혀·턱 | EchoSpeech, HPSpeech, EchoNose, SoniSpeech, SoniGAN | EarCommand, EarSSR, ReHEarSSE, HEar-ID | CanalScan(폰을 귀에 댐) | EchoSpeech WER 4.5%, ReHEarSSE 1000단어 73.3% |
| 수어 | SpellRing(지문자) | SonicASL | — | SonicASL 단어 93.8%(사용자별), 56.3%(사용자 독립) |
| 몸 자세·활동·머리 | PoseSonic, ActSonic, WristSonic | CochlearMotion(머리 제스처) | Tarzia, Amazon Echo(존재) | PoseSonic 6.17 cm, ActSonic macro-F1 93.4% |
| 연속 손 자세 | EchoWrist, Ring-a-Pose, WatchHand | — | Beyond-Voice, SonicHand, RAM-Hand | EchoWrist 4.81 mm, WatchHand 재착용 7.87 mm, RAM-Hand 10.71 mm |
| 손가락 추적·손 제스처·힘 | Grab-n-Go, EchoForce(쥐는 힘) | MAF, EarHover, HandLeap | FingerIO, LLAP, Strata, VSkin, RobuCIR, UltraGesture, AudioGest, SoundWave, SonarID, SonarWatch, SonarSelect, FingerPing, dCIR + LLM, 기기 추적 CAT·MilliSonic | LLAP 1D 3.5 mm, SonarSelect 3.45–4.53 mm, RobuCIR 98.4% |
| 먹기·마시기 | ActSonic, MunchSonic, EchoGuide | — | — | MunchSonic macro-F1 0.935 |
| 숨쉬기·수면 | EchoNose(숨 제스처) | — | ApneaApp, Second Chance, BreathJunior, C-FMCW, SonarBeat, BreathListener, RespTracker, MultiResp, LoEar, Sci Rep 2026 | ApneaApp AHI ICC 0.9816, Sci Rep 2026 MAE 0.29 brpm |
| 심박 | — | APG(이어버드) | ACG, Zhang 2020, Commun Biol 2021, Globecom 2021, DF-Sense, Sensors 2024 | APG 심박 중앙 오차 3.21%(153명), Commun Biol R–R 28 ms |
| 폐기능·귀 진단 | — | — | SpiroSonic, PTEase, SpiroSense, 중이 삼출액 | SpiroSonic 흉벽 ≤3.8 mm, 중이 AUC 0.898 |
| 인증 | SonicID | EarEcho, EarDynamic, HEar-ID | EchoPrint | SonicID TPR 97.2%, EchoPrint BAC 93.75% |
| 물질·재실 | — | ASLiquid(이어폰) | Akte-Liquid, ASLiquid+, IceSound, Shih & Rowe(재실 인원) | ASLiquid F1 95–99.25% |

## 계열마다 레시피 하나가 굳어 있다

같은 계열 안에서는 기기, 신호, 처리가 거의 바뀌지 않는다. 그래서 계열별로 묶어 적는다. 표의 S는 스피커, M은 마이크다. 수치는 각 논문이 내세운 대표값이고, 평가 방식이 달라 계열 사이에서 바로 견줄 수는 없다.

### Cornell 계보는 600샘플 C-FMCW와 차분을 4년째 쓴다

EarIO 이후 처리는 거의 그대로다. 받은 소리를 송신 대역으로 거르고(대개 5차 Butterworth) 송신 처프와 상관한 뒤, 처프 주기마다 한 줄로 접는다. 직접 경로를 영점으로 두고(EarIO·Ring-a-Pose·WatchHand는 600샘플 중 300번째), 가까운 거리만 잘라 이웃 줄과의 차분을 낸다 ([EarIO](https://galaxywalk.github.io/files/imwut22eario.pdf), [Ring-a-Pose](https://arxiv.org/html/2404.12980)). 위상을 쓰는 논문은 없다. 얼굴·말에는 차분만, 모양이 중요한 손·몸에는 원본과 차분을, 신원에는 원본만 넣는다. 맞춤 기기는 Teensy·nRF52840에 S·M 두 쌍을 달고, 50 kHz에서 18–21 kHz와 21.5–24.5 kHz로 두 스피커를 가른다. 스피커가 하나면 20–24 kHz를 주로 쓴다. 2025–26년에는 표본화율을 올려 대역을 넓힌다(EchoForce 20–29 kHz @96 kHz, SoniSpeech 18–28·29–39 kHz @100 kHz) ([EchoForce](https://arxiv.org/html/2507.20437), [SoniSpeech](https://arxiv.org/html/2608.00803)). 몸통은 ResNet-18이 기본이고, FastViT가 EchoForce와 WatchHand에 들어왔다. 정답은 ARKit, BlazePose, MediaPipe Hands다. 적응은 새 사용자를 뺀 사전학습 뒤 미세조정이고, 재착용에는 거리 축 이동 증강을 쓴다. 대역 밖 소음(음악·카페·거리)은 거의 해가 없고, 감지 범위 안의 움직임(걷기, 다른 팔, 물체)과 손 모양 변화가 성능을 떨어뜨린다 ([WatchHand](https://arxiv.org/html/2602.21610), [EchoWrist](https://arxiv.org/html/2401.17409)).

| 시스템 · ID | 기기 · S/M | 신호 · 대역 · 프레임 | 처리 · 표현 | 모델 · 출력 | 정답 | 보정 · 견고성 · 대표 수치 |
|---|---|---|---|---|---|---|
| [EarIO](https://galaxywalk.github.io/files/imwut22eario.pdf) · DOI 10.1145/3534621 (IMWUT 2022) | 개조 이어폰, 양쪽 1S+1M | FMCW 16–20 kHz, 44.1 kHz·512 또는 50 kHz·600 | 차분만, ±19.75 cm | ResNet-34 → blendshape 52 | iPhone TrueDepth + ARKit, 박수 동기 | 차분이 재착용에 가장 강함. 세션 사이 MAE FMCW 19.2 < GSM CIR 23.8 < 처프 27.7 |
| [EchoSpeech](https://ruidongzhang.com/files/papers/EchoSpeech_authors_version.pdf) · DOI 10.1145/3544548.3580801 (CHI 2023) | 안경 아래 테, S 2개와 M 2개를 반대편에 | 18–21 + 21.5–24.5 kHz, 50 kHz·600(12 ms) | 차분만, 4경로 = 4채널 | ResNet-18 + CTC | 지시문 + 박수 | WER 4.5%. 배치만 바꿔 <50% → 98% |
| [PoseSonic](https://par.nsf.gov/servlets/purl/10492772) · DOI 10.1145/3610895 (IMWUT 2023) | 안경 경첩, 2S+2M 아래 향 | 같은 이중 대역, 900샘플(18 ms) | 원본 + \|차분\| 두 갈래 | ResNet-18 둘, 교사–학생 | BlazePose GHUM 3D 영상 교사 | 상체 9관절 6.17 cm(실험실, 새 사용자) |
| [HPSpeech](https://par.nsf.gov/servlets/purl/10492815) · DOI 10.1145/3594738.3611365 (ISWC 2023) | 헤드폰 내장 스피커 + 덧붙인 마이크 | 20–24 kHz를 음악에 섞음 | 에코 프로파일 | ResNet-18 분류 | 지시 | 8명령 >90%. 내장 마이크는 API가 없어 못 씀 |
| [EchoNose](https://par.nsf.gov/servlets/purl/10492814) · DOI 10.1145/3594738.3611358 (ISWC 2023) | 안경, 콧구멍 아래 1S+1M | 20–24 kHz, 50 kHz | 차분 | ResNet-18 분류 | 지시 | 말·혀·숨 16제스처 93.7% |
| [EyeEcho](https://arxiv.org/html/2402.12388) · arXiv 2402.12388 (CHI 2024) | 안경 다리, 2S+2M 볼 향 | 16–20 kHz, 50 kHz·600 | 차분, 10.2 cm | ResNet-34 → blendshape | ARKit | MAE 22.9. 자리 비교 18.0·18.0·22.8 |
| [GazeTrak](https://arxiv.org/html/2402.14634) · arXiv 2402.14634 (MobiCom 2024) | 안경, 2S+8M | 이중 대역, 600 | 70칸 중 60칸 무작위 자르기 | ResNet-18 MSE | 화면 지시점(Tobii로 검증) | 세션 사이 4.9°. 세기 20%로 운용(2%도 비슷) |
| [EchoWrist](https://arxiv.org/html/2401.17409) · arXiv 2401.17409 (CHI 2024) | 손목밴드, 손등 쪽·손바닥 쪽 2쌍 | 20–24 kHz(본문에 18–21 kHz도), 두 스피커 같은 신호 | 원본 + 차분, 72칸 24.6 cm | ResNet-18 MSE → 20관절 60값 | 웹캠 + MediaPipe, 손가락 튕기기 동기 | 미세조정 4.81 mm, 새 사용자 12.2 mm, 0.5세션 6.92 mm. ±11칸 이동 증강 |
| [Ring-a-Pose](https://arxiv.org/html/2404.12980) · arXiv 2404.12980 (IMWUT 2024) | 반지, 1S+1M 5.4 mm 간격 | 20–24 kHz, 50 kHz·600 | 원본 + 차분, 54칸, 채널별 정규화 | ResNet-18(ResNet+RNN과 같고 transformer보다 나음) | MacBook 카메라 + MediaPipe | 새 사용자 14.1 mm, 미세조정 10.3 mm. 46%가 소리를 들음 |
| [ActSonic](https://arxiv.org/html/2404.13924) · arXiv 2404.13924 (IMWUT 2024) | 안경 2S+2M | 이중 대역, 600 | 차분, 300칸(약 1 m) | 가림 복원 자기지도 사전학습 | 초 단위 주석 | 27활동 macro-F1 93.4% |
| [SonicID](https://arxiv.org/html/2406.08273) · arXiv 2406.08273 (IMWUT 2024) | 안경 2S+2M | 이중 대역 | 원본만(얼굴 모양) | ResNet-18 분류 | 신원 | TPR 97.2%, FPR 4.6% |
| [MunchSonic](https://arxiv.org/html/2405.21004) · arXiv 2405.21004 · [EchoGuide](https://arxiv.org/html/2406.10750) · arXiv 2406.10750 (ISWC 2024) | 안경 경첩 | 이중 대역, 600 | 차분, 150칸 | MobileNetV2 · ActSonic 모델 | 주석 | 먹기 macro-F1 0.935 · 녹화 68% 줄임 |
| [WristSonic](https://arxiv.org/html/2411.08217) · arXiv 2411.08217 (2024 선공개) | 워치 모양 껍데기, 보드 둘(바깥·몸 쪽) | 이중 대역, 83 fps | 차분 4채널, 200칸 | transformer | 주석 | 21동작 macro-F1 93.08%. 90° 기울임이 가장 유익 |
| [SpellRing](https://arxiv.org/html/2502.10830) · arXiv 2502.10830 (CHI 2025) | 반지 1S+1M + IMU | 20–24 kHz, 감지 빠르기 6배(범위 2.06 m → 34.3 cm) | 차분 | ResNet-18 + 자이로 CNN, CTC | 타임스탬프 | 6배로 57.93% → 86.03%. 구 단위 WER 0.099 |
| [EchoForce](https://arxiv.org/html/2507.20437) · arXiv 2507.20437 (ISWC 2025) | 손목밴드 1S+1M, 45° 팔뚝 향 | 20–29 kHz @96 kHz, 160 fps, 칸 1.79 mm | 차분, 78칸 | FastViT MSE | 악력계 화면 영상 | 오류율 9.08%(사용자별), 12.3%(독립) |
| [Grab-n-Go](https://arxiv.org/html/2508.11620) · arXiv 2508.11620 (IMWUT 2025) | 손목밴드 2쌍 | 이중 대역, 12 ms | 원본 4 + 차분 4 | ResNet-18 분류 | 컵 두드림 동기 | 30 미세제스처 92.0% |
| [WatchHand](https://arxiv.org/html/2602.21610) · arXiv 2602.21610 (CHI 2026) | 기성 Galaxy Watch 7·Xiaomi Watch 2 Pro·Pixel Watch 3, 내장 1S+1M | 18–21 kHz, 48 kHz·600(12.5 ms), UNPROCESSED | 5차 Butterworth → 상관 → 봉우리 맞춤(전체 + 미끄럼 창) → 중앙값 → \|P[t]\|−\|P[t−1]\|, 60칸 21.42 cm | FastViT-T12, MSE + 0.1·속도 | MacBook 웹캠 30 fps + MediaPipe, 손바닥 정면 | 재착용 7.87 mm, 세션 안 6.02, 새 사용자 14.88. ±5칸 이동·마스킹. 기기 효과 유의하지 않음 |
| [SoniSpeech](https://arxiv.org/html/2608.00803) · arXiv 2608.00803 · [SoniGAN](https://arxiv.org/html/2610.15671) · arXiv 2610.15671 (2026 선공개) | 안경 2S + PDM 마이크 2 | 18–28 + 29–39 kHz @100 kHz, 5 ms | 차분 4채널 · 글에서 합성한 프로파일 | ResNet-34 CTC · ResNet-18 + Transformer | 한 사람 34 h | WER 26.3%. 합성 104 h 사전학습으로 25.9 → 23.6% |

### 다른 머리 착용 기기는 거리 대신 전달함수와 Doppler를 본다

Cornell 밖 머리 착용 시스템 가운데 거리 에코 프로파일과 프레임 차분을 쓰는 것은 찾지 못했다. 외이도 계열은 주파수 쪽 전달함수(EarCommand, EARFace, ECTF, EarSSR)나 OFDM과 AR 계수(ReHEarSSE, HEar-ID, FIT2025)를 쓴다. 대상인 외이도(2–3 cm)가 거리 분해능보다 작기 때문이다. 바깥 계열은 순음이나 코드로 채널을 잰다(FaceListener, MAF, EarHover, SonicASL). 배경은 이웃 줄 차분이 아니라 착용 직후 기준과의 차로 지운다. EarCommand는 착용 직후 40 ms 처프로 잰 정적 외이도 응답을 빼고, ECTF는 기본 표정에서 잰 기준으로 보정한다. 그래서 멈춘 상태의 정보가 남는다 ([EarCommand](https://dl.acm.org/doi/10.1145/3534613), [ECTF](https://dl.acm.org/doi/10.1145/3341163.3347747)). 상용 이어버드는 귓속 마이크의 원시 신호를 내주지 않아, 마이크를 떼어 다시 잇거나(ReHEarSSE) 덧붙였다(EarCommand, EarSSR, EarHover). 성능 무늬는 모두 같다. 사용자 독립은 무너지고(SonicASL 56.3%, EarCommand WER 87.29%, ReHEarSSE 19.5%, EarHover 약 48%), 짧은 개인 자료로 돌아온다(EarCommand 클래스당 24개로 WER 11.36%, ReHEarSSE 60단어로 78.5%, EARFace 2분 적응이 5분 사용자별과 비슷) ([SonicASL](https://dl.acm.org/doi/10.1145/3463519), [ReHEarSSE](https://dl.acm.org/doi/10.1145/3613904.3642095), [EARFace](https://www.cse.psu.edu/~mkg31/papers/earface.pdf)).

| 시스템 · ID | 기기 · S/M | 신호 · 대역 · 프레임 | 처리 · 표현 | 모델 · 출력 | 정답 | 보정 · 견고성 · 대표 수치 |
|---|---|---|---|---|---|---|
| [SonicASL](https://dl.acm.org/doi/10.1145/3463519) · DOI 10.1145/3463519 (IMWUT 2021) | 헤드폰에 붙인 바깥 향 모듈(마이크 + 스피커 1 cm), 폰 | 20 kHz 순음, STFT 8,192점 | Doppler 스펙트로그램, 반송파 띠 제거·Wiener·Gaussian, 224×224 | VGG16식 CNN + BiLSTM + CTC | 오디오 신호에 맞춘 지시 | 단어 93.8%·문장 90.6%(사용자별), 56.3%(독립). 175 cm에서 0.5 W 23% 대 1 W 72% |
| [EarCommand](https://dl.acm.org/doi/10.1145/3534613) · DOI 10.1145/3534613 (IMWUT 2022) | 이어폰 스피커 + 귓속 마이크, Pixel 3A | FMCW 16–20 kHz, 48 kHz | 수신/송신 주파수 비, 착용 직후 정적 응답 빼기, Wiener | 1D CNN + BiLSTM + CTC | 지시 | 단어 WER 10.02%. 새 사용자 87.29% → 클래스당 24개 11.36%. 재착용 6.67 대 14.44% |
| [FaceListener](https://par.nsf.gov/servlets/purl/10335060) · DOI 10.1109/IPSN54338.2022.00019 (IPSN 2022) | 기성 유선 헤드폰, 한쪽 송신·반대쪽 스피커를 마이크로 | BPSK 코드 다중 톤 18–21 kHz | 최소제곱 채널 추정 | LSTM, 30 keypoint 회귀 + 표정 분류 | MediaPipe Face Mesh 교사 | >80%(사용자별), 약 70%(독립). 라벨만으로는 수렴하지 않음 |
| [EARFace](https://www.cse.psu.edu/~mkg31/papers/earface.pdf) · DOI 10.1145/3614438 (ACM TIOT 2023) | 맞춤 이어팁, 초음파 스피커 + MEMS 마이크, 양쪽 | 16–40 kHz sweep, 80 kHz, 0.05 s | CFR = FFT(수신)/FFT(송신), LFCC 100 | spectral + temporal transformer → 랜드마크 51 | 폰 전면 카메라 + OpenFace 2.0 | MAE 1.83 mm. 2분 적응 ≈ 5분 사용자별 |
| [EarSSR](https://chaof94.github.io/Papers/EarSSR.pdf) · DOI 10.1109/TMC.2024.3356719 (TMC 2024) | 이어폰 + 붙인 마이크 칩, 스펀지 | FMCW 16–22 kHz, 10 ms, PN 앞머리 | 전달함수 + 정적 경로 뺀 진폭 변화의 CWT | MobileNetV3 + GRU | 지시 | 단어 약 93%. 스펀지 없으면 93 → 72% |
| [ReHEarSSE](https://dl.acm.org/doi/10.1145/3613904.3642095) · DOI 10.1145/3613904.3642095 (CHI 2024) | ANC 이어버드 귓속 마이크 재배선, Pixel 3a | OFDM 17.5–23 kHz, 2046샘플(43 ms), PAPR 52% 낮춤 | 상관 + 위상 차 최소로 ≤4샘플 정렬, 2차 미분, AR 200 | TCN + CTC 사전 순위 | 화면 + 스페이스바 | 같은 대역·길이에서 OFDM 89.7% 대 FMCW 82.5%. 새 사용자 19.5% → 60단어 78.5% |
| [HEar-ID](https://arxiv.org/html/2512.16518v1) · arXiv 2512.16518, DOI 10.1145/3714394.3754429 (UbiComp Companion 2025) | 같은 하드웨어 | OFDM 17.5–23 kHz | ±25샘플 미세 정렬, AR | TCN + BiGRU, 대조 학습 | — | 인증 TPR 81.76%, FPR 3.2% |
| [FIT2025](https://info.ipsj.or.jp/award/9faeag0000004f1r-att/CJ-002_1.pdf) · DOI·arXiv 없음 (FIT 2025) | 조금 고친 이어버드, 오른쪽 | 미보고 | AR + 단구간 에너지 | ResNet18 + GRU → blendshape 52 | MediaPipe blendshape | 2명, 수치는 그림에만 |
| [ECTF](https://dl.acm.org/doi/10.1145/3341163.3347747) · DOI 10.1145/3341163.3347747 (ISWC 2019) | 이어폰 + MEMS 마이크, 양쪽, 96 kHz | swept sine 18–48 kHz | 역필터 전달함수, 기본 표정 기준 보정, LFCC | SVM | 지시 | 21동작 40.2% → 보정 후 62.5% |
| [MAF](https://dl.acm.org/doi/10.1145/3613904.3642437) · DOI 10.1145/3613904.3642437 (CHI 2024) | 골전도 이어폰 + 얼굴에 붙인 마이크 | 18 kHz 순음(FMCW는 응답 출렁임·과도음 때문에 버림) | STFT, KL 분할 | CRNN | 영상 감독 | 10제스처 92.8%(교차 참가자) |
| [EarHover](https://dl.acm.org/doi/10.1145/3654777.3676367) · DOI 10.1145/3654777.3676367 (UIST 2024) | 이어폰 + 바깥 마이크 추가 | 20 kHz 사인 누설음 | Doppler 스펙트로그램 | 2단 CNN | 수작업 주석 | F 79.7–85.9%(사용자별), 약 48%(LOSO). 20 kHz를 들은 사람 없음 |
| [EarEcho](https://dl.acm.org/doi/10.1145/3351239) · DOI 10.1145/3351239 (IMWUT 2019) | 시제품 이어피스 | 가청 재생음 + 6 kHz 이하 처프 | 전달함수, 스피커 응답 보상 | SVM | 신원 | 정밀도 95.16% |
| [EarDynamic](https://dl.acm.org/doi/10.1145/3448098) · DOI 10.1145/3448098 (IMWUT 2021) | 귓속 시제품 + IMU | 16–23 kHz 처프 | 음소별 외이도 변형 | boosting | 신원 | 정확도 93.04% |
| [CanalScan](https://yetongcao.github.io/files/CanalScanTMC-Final.pdf) · DOI 10.1109/INFOCOM42981.2021.9488852 (INFOCOM 2021) | 폰 수화부를 귀에 | 16 kHz 순음 | 반사 포락선 통계 | Random Forest | 지시 | 재현율 94.84% |
| [CochlearMotion](https://aisel.aisnet.org/hicss-58/in/hci/2) · DOI 없음 (HICSS-58) · [HandLeap](https://www.bohrium.com/paper-details/handleap-towards-contact-free-gesture-interaction-with-earphones-via-acoustic-sensing/1114322869342437382-62916) · DOI 10.1007/s42486-024-00180-x | 상용 이어버드·이어폰 | 초록만 | — | — | — | 머리 제스처 6종 >95%(CochlearMotion 초록) |

### 폰·스피커·손목 추적은 복소 CIR과 위상으로 mm를 만든다

이 갈래는 신호로 넷이다. OFDM(FingerIO), CW 다중 톤 위상(LLAP와 이를 고친 SonarSelect), 훈련열·ZC로 잰 CIR(Strata, VSkin, RobuCIR, SonicHand), FMCW(MilliSonic, Beyond-Voice, RAM-Hand)다. 순음 Doppler(SoundWave, AudioGest)가 따로 있다. 거의 모두 48 kHz에서 17–23 kHz 안의 2–6 kHz를 쓴다. 칸 간격은 c/2fs = 3.5–3.6 mm지만 대역이 정하는 분해능은 c/2B = 3–6 cm이고, mm 정확도는 모두 위상에서 나온다 ([Beyond-Voice](https://arxiv.org/html/2306.17477v2), [VSkin](https://samsonsjarkal.github.io/KeSun/files/mobicom18_vskin.pdf)). 정적 성분은 복소 CIR 차분(VSkin, RobuCIR, SonicHand)이나 정적 벡터 추정 빼기(LLAP의 LEVD)로 지운다. 우리 실수 상관 차분에 가장 가까운 것은 Beyond-Voice(상관 차분 뒤 ReLU)다. 연속 3D 손 자세는 손목 밖 기기에서도 10.7–23 mm다 ([RAM-Hand](https://wenjunjiang.github.io/SenSys2025Shiyang.pdf), [SonicHand](https://engineering.purdue.edu/~lusu/papers/TOSN2024.pdf)).

| 시스템 · ID | 기기 · S/M | 신호 · 대역 · 프레임 | 처리 · 표현 | 모델 · 출력 | 정답 | 보정 · 견고성 · 대표 수치 |
|---|---|---|---|---|---|---|
| [FingerIO](https://fingerio.cs.washington.edu/fingerio.pdf) · DOI 10.1145/2858036.2858580 (CHI 2016) | Galaxy S4 스피커 + 마이크 2. 워치 시제품은 외부 마이크 2 | OFDM 18–20 kHz, 84샘플 + 무음 200(5.92 ms), 48 kHz | 프로파일 빼기, 직접 경로 진폭 15% 상대 문턱, FFT 선형 위상으로 시작점·분수 지연 | 학습 없음 | — | 2D 8 mm(폰), 1.2 cm(워치) |
| [LLAP](https://samsonsjarkal.github.io/KeSun/files/mobicom16_ultrasound.pdf) · DOI 10.1145/2973750.2973764 (MobiCom 2016) | Galaxy S5 마이크 2 | CW 다중 톤 17–23 kHz, 350 Hz 간격 | I/Q(CIC), LEVD로 정적 벡터 빼기, 위상 → 경로 길이 | 학습 없음 | — | 1D 3.5 mm, 2D 4.57 mm. Android는 10–20 ms마다 소리 전달 |
| [Strata](https://doi.org/10.1145/3081333.3081356) · DOI 10.1145/3081333.3081356 (MobiSys 2017) | Galaxy S4 | 26-bit GSM 훈련열 CIR | 손가락 탭을 골라 위상 추적 | 학습 없음 | — | 원문 미확인 |
| [VSkin](https://samsonsjarkal.github.io/KeSun/files/mobicom18_vskin.pdf) · DOI 10.1145/3241539.3241568 (MobiCom 2018) | Galaxy S5 뒷면 스피커 + 위·아래 마이크 | ZC 127 → 1024점 보간, B 5.953 kHz, 21.3 ms | IR 시간 차분, 차분이 가장 큰 지연 고르기, EKF, 곡률 위상 | 학습 없음 | — | 이동 3.59 mm, 탭 99.65%. 지연 보정은 처음 213 ms에 한 번 |
| [RobuCIR](https://shenjiaxing.win/pdf/wang_push_2020.pdf) · DOI 10.1109/TMC.2020.3032278 (INFOCOM 2020, TMC 2022) | Galaxy S9+·S7 Edge·Nexus 5 | TSC ×12, 18/20/22 kHz 도약, 6.5 ms | LS CIR 크기·위상, 복소 차분, 위상 오프셋 제거, min–max | CNN + LSTM, 반송파 셋을 RGB처럼 | 지시 | 15제스처 98.4%. 한 주파수만 쓰면 81%·78.2% |
| [UltraGesture](https://dblp.org/rec/journals/tmc/LingDLLWG22.html) · DOI 미확인 (SECON 2018, TMC 2022) | 상용 | CIR | CIR 크기 | DNN | — | 약 97%(2차 출처) |
| [MilliSonic](https://arxiv.org/abs/1901.06601) · arXiv 1901.06601 (CHI 2019) | 폰 스피커 → 마이크 4개 배열(기기 추적) | FMCW 17.5–23.5 kHz, 45 ms | FMCW 순간 위상, 동적 협대역 필터 | 학습 없음 | — | 1D 0.7 mm. 5초 녹음으로 시계 표류 보정 |
| [Beyond-Voice](https://arxiv.org/html/2306.17477v2) · arXiv 2306.17477 (IPSN 2024) | Echo Dot 같은 7마이크 배열 + 스피커 | FMCW 17–20 kHz, 512샘플, 48 kHz | xCORR 차분 + ReLU, 직접 경로는 한 번 | CNN + LSTM → 21관절 | 확인 못 함 | 16.47 mm(독립) → 10.36 mm(적응). 초음파 30 dB에서 20.23 mm |
| [SonicHand](https://engineering.purdue.edu/~lusu/papers/TOSN2024.pdf) · DOI 10.1145/3677122 (TOSN 2024) | Pixel 2 스테레오 | 20 kHz 순음(Doppler) + ZC 16.5–19.5 kHz | 프레임마다 LOS를 첫 칸으로, 복소 CIR 차분, 곡률 위상 | CNN + attention + LSTM → 관절 회전 + 골격 | Leap Motion 100 Hz | 22.87 mm, 한 손에 폰 18.64 mm. 적대적 학습 |
| [RAM-Hand](https://wenjunjiang.github.io/SenSys2025Shiyang.pdf) · DOI 10.1145/3715014.3722055 (SenSys 2025) | 맞춤 L자 16마이크 + 스피커 | FMCW 16–22 kHz, Hanning, 20 ms, 44.1 kHz | range FFT 차분, MVDR 거리–방위–고도 큐브 | TimeSformer식, 대조 학습 | 확인 못 함 | 관절 평균 10.71 mm |
| [SonarID](https://doi.org/10.1145/3491102.3501935) · DOI 10.1145/3491102.3501935 (CHI 2022) | 기성 워치(모델 미확인) | ZC(학위논문 요약 기준) | 소나 지문 | 딥러닝 분류 | 지시 | 엄지·검지·중지 식별 최대 93.7% |
| [SonarWatch](https://arxiv.org/abs/2408.12689) · arXiv 2408.12689 (2024 선공개) | Mi Watch 껍데기 + 외부 녹음기, 재생은 폰 | 오름·내림 처프 16.5–20 kHz, 4096샘플 | STFT 크기 + IMU | 분류 | 지시 | 12분류 93.7%. 12명 중 11명은 소리를 못 들음 |
| [SonarSelect](https://arxiv.org/html/2503.02308v1) · arXiv 2503.02308, DOI 10.1145/3706598.3714308 (CHI 2025) | Galaxy Watch 6, 마이크 1 | LLAP 각색(CW 위상), 대역 미기재 | LEVD를 0–10 cm에 맞춤, One-Euro 필터 | 학습 없음 | — | 3.45 mm(0–5 cm), 4.53 mm(5–10 cm), 그 밖은 급히 나빠짐 |
| [FingerPing](https://doi.org/10.1145/3173574.3174011) · DOI 10.1145/3173574.3174011 (CHI 2018) | 엄지 반지 트랜스듀서 + 수신기 4 | 몸으로 전하는 처프 20 Hz–6 kHz | 주파수 응답 | 분류 | 지시 | 93.77%, 95.64% |
| [AudioGest](https://doi.org/10.1145/2971648.2971736) · DOI 10.1145/2971648.2971736 · [SoundWave](https://doi.org/10.1145/2207676.2208331) · DOI 10.1145/2207676.2208331 | 상용 기기 1쌍 | 비가청 순음 Doppler | Doppler·다중경로 | 학습 없음 | — | 6제스처 최대 96%(AudioGest) |
| [CAT](https://doi.org/10.1145/2973750.2973755) · DOI 10.1145/2973750.2973755 (MobiCom 2016) | 여러 스피커 → 폰(기기 추적) | 분산 FMCW + Doppler + IMU | — | — | — | mm급(초록) |
| [dCIR + LLM](https://arxiv.org/abs/2511.07085) · arXiv 2511.07085 (2025 선공개) | 폰 | GSM 26-bit, 50-bit 프레임 ×12 = 600샘플 | 차분 CIR 영상 | LLM 소수샷 | — | 고전 ML과 비슷 |

### 생리 연구는 긴 적분과 주파수 모으기로 sub-mm를 잡는다

숨·심박 연구는 작고 느린 움직임을 공통 틀로 잡는다. 움직임 대역 에너지 대 잡음으로 거리 칸을 고르고, 그 칸의 위상을 시간에 따라 쫓는다. 여러 주파수·부반송파·마이크를 SNR 가중으로 코히어런트하게 더하고, 숨(약 0.2–1.1 Hz)과 심박(60–150 CPM)을 대역으로 가른다 ([BreathJunior](https://homes.cs.washington.edu/~gshyam/Papers/whitenoise.pdf), [Commun Biol 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7943557)). 크기 변화만 쓰는 것은 존재 감지처럼 큰 움직임용이었다 ([Tarzia](https://users.cs.northwestern.edu/~pdinda/Papers/ubicomp09.pdf)). BreathJunior와 SpiroSense는 상관만으로는 샘플 간격보다 작은 움직임을 못 잡는다고 적었다 ([SpiroSense](https://doi.org/10.1145/3749489)). 대부분 학습 없이 규칙으로 끝내고, 정답은 PSG, ECG, spirometer다. 몸에 찬 기기는 이어버드뿐이었고(APG, EarEcho, EarDynamic), 스마트워치로 생리 상태를 능동 음향으로 잰 1차 문헌은 찾지 못했다.

| 시스템 · ID | 기기 · S/M | 신호 · 대역 · 프레임 | 처리 · 표현 | 모델 | 정답 | 보정 · 견고성 · 대표 수치 |
|---|---|---|---|---|---|---|
| [ApneaApp](https://homes.cs.washington.edu/~gshyam/Papers/apneaapp.pdf) · DOI 10.1145/2742647.2742674 (MobiSys 2015) | Galaxy S4 등, 폰 스피커 + 떨어진 마이크 | FMCW 18–20 kHz, 10.75 ms, 48 kHz | 처프 10개에 걸친 FFT로 칸 폭 9.37 Hz, 0.2–0.3 Hz 봉우리로 칸 고르기 | 규칙 | PSG | AHI ICC 0.9816(37명). 붙은 S·M로 동시에 내고 받으면 진폭이 흔들림 |
| [Second Chance](https://doi.org/10.1126/scitranslmed.aau8914) · DOI 10.1126/scitranslmed.aau8914 (Sci Transl Med 2019) | 폰 | 초록만 | — | — | 감독 주사 시설 | 중추성 무호흡 민감도 96%·특이도 98%(209명) |
| [BreathJunior](https://homes.cs.washington.edu/~gshyam/Papers/whitenoise.pdf) · DOI 10.1145/3300061.3345453 (MobiCom 2019) | 7마이크 배열 + 스피커 | 백색잡음 0.2 s 블록 + cyclic prefix → 3 kHz FMCW 다섯 조각(6–21 kHz) | 칸 위상 추적, 조각별 SNR 가중, 빔포밍 | 규칙 | 신생아 시뮬레이터, NICU 모니터 | NICU 상관 0.938. 마이크 12 kHz 근처 5–10 dB 골 |
| [C-FMCW](https://doi.org/10.1145/3161188) · DOI 10.1145/3161188 (IMWUT 2018) | 붙여 둔 스피커 + 마이크 | 상관 기반 FMCW(우리 처리의 원조) | SFO 보정, 자기 간섭 기준 정렬, 자기상관 | — | — | 호흡 중앙 오차 <0.35 bpm |
| [LoEar](https://doi.org/10.1145/3550293) · DOI 10.1145/3550293 (IMWUT 2022) | 1S + 1M | 다중 부반송파 CFR | Continuous-MUSIC, Carrierforming | — | — | 호흡 2 → 7 m, 심박 1.2 → 6.5 m |
| [Sci Rep 2026](https://doi.org/10.1038/s41598-026-69919-z) · DOI 10.1038/s41598-026-69919-z | 폰 | 18–20 kHz 순음, I/Q | 자기상관, 일관성 점수 | — | PSG 벨트 | ICC 0.97, MAE 0.29 brpm(80명) |
| [Commun Biol 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7943557) · DOI 10.1038/s42003-021-01824-9 | 스마트 스피커 시제품, 마이크 배열, 48 kHz | FMCW 18–22 kHz, 50 ms 블록, 10 ms 이동 DFT | 1 m 밖 에코 지움, 마이크 × 주파수 복소 가중을 정답 없이 처음 30 s로 최적화 | 학습 없음 | ECG(Polar H10 등) | R–R 중앙 오차 28 ms(건강), 30 ms(환자). 18–22 kHz 응답이 고르지 않음 |
| [APG](https://storage.googleapis.com/gweb-research2023-media/pubtools/7284.pdf) · DOI 10.1145/3570361.3613281 (MobiCom 2023) | 기성 ANC 이어버드, ANC 스피커 + 귓속 마이크 | 30 kHz 넘는 다중 톤 CW | coherent 복조, 착용마다 PAR 최고 톤 고르기, BSS | 모델 기반 | Polar H10, Nonin 산소포화도계 | 심박 중앙 오차 3.21%(153명) |
| [SpiroSonic](https://par.nsf.gov/servlets/purl/10268386) · DOI 10.1145/3372224.3419209 (MobiCom 2020) | Galaxy S8 등 손에 든 폰, 가슴 5–20 cm | CW 12톤 17–22.5 kHz, AGC 끔 | 위상 → 변위, I/Q 원호를 구간마다 맞춰 중심 옮기기 | Bayesian NN 회귀 | spirometer, 레이저 거리계 | 흉벽 ≤3.8 mm. AGC를 꺼도 SNR >40 dB |
| [PTEase](https://par.nsf.gov/servlets/purl/10457362) · DOI 10.1145/3581791.3596854 (MobiSys 2023) | 폰 + 3D 프린트 통로 | 2 ms 펄스, 6 kHz 이하 | deconvolution, 품질 검사 | 자기지도 + 다중 과제 | 임상 | 질병 예측 75%(182명) |
| [SpiroSense](https://doi.org/10.1145/3749489) · DOI 10.1145/3749489 (IMWUT 2025) | 폰 + 부착물 | 초음파 | 상관 대신 위상 특징 | 학습 | 59명 | 분해능 11.9 → 0.148 L/s |
| [중이 삼출액](https://par.nsf.gov/servlets/purl/10095249) · DOI 10.1126/scitranslmed.aav1102 (Sci Transl Med 2019) | iPhone 5s·Galaxy S6 + 종이 깔때기 | 가청 처프 1.8–4.4 kHz, 150 ms | dip 모양 | logistic regression | 수술 소견 | AUC 0.898 |
| [EchoPrint](https://par.nsf.gov/servlets/purl/10095814) · DOI 10.1145/3241539.3241575 (MobiCom 2018) | 수화부 스피커 + 위 마이크 + 전면 카메라 | 1 ms 처프 16–22 kHz, Hanning | 조용한 곳에서 녹음한 직접 경로 템플릿, 가운데 15샘플만 상관, 카메라 거리로 에코 위치 보정 | CNN 특징 + SVM | 신원 | BAC 93.75%(62명) |
| 그 밖(대부분 초록만) | — | — | — | — | — | [ACG](https://doi.org/10.1109/infocom.2018.8485978) 10.1109/infocom.2018.8485978(심박 0.6 bpm), [Zhang 2020](https://doi.org/10.1145/3432237) 10.1145/3432237(박동 간격 13.28 ms), [DF-Sense](https://doi.org/10.1145/3581791.3596867) 10.1145/3581791.3596867, [Globecom 2021](https://doi.org/10.1109/globecom46510.2021.9685922) 10.1109/globecom46510.2021.9685922, [Sensors 2024](https://doi.org/10.3390/s24020382) 10.3390/s24020382, [SonarBeat](https://doi.org/10.1109/icccn.2017.8038412) 10.1109/icccn.2017.8038412, [BreathListener](https://doi.org/10.1145/3307334.3326074) 10.1145/3307334.3326074, [RespTracker](https://doi.org/10.1109/infocom42981.2021.9488881) 10.1109/infocom42981.2021.9488881, [MultiResp](https://doi.org/10.1109/tmc.2023.3279976) 10.1109/tmc.2023.3279976, [BlinkListener](https://doi.org/10.1145/3463521) 10.1145/3463521, [Tarzia](https://users.cs.northwestern.edu/~pdinda/Papers/ubicomp09.pdf) 10.1145/1620545.1620559, [Shih & Rowe](https://doi.org/10.1145/2735960.2735969) 10.1145/2735960.2735969, [Akte-Liquid](https://doi.org/10.1145/3551640) 10.1145/3551640, [ASLiquid](https://doi.org/10.1145/3666025.3699321) 10.1145/3666025.3699321, [ASLiquid+](https://doi.org/10.1109/tmc.2026.3711400) 10.1109/tmc.2026.3711400, [IceSound](https://doi.org/10.1145/3749543) 10.1145/3749543, [Amazon Echo](https://www.amazon.science/blog/the-science-behind-ultrasonic-motion-sensing-for-echo)(블로그) |

## 우리는 계보를 세 곳에서 벗어나고, 계보 밖과는 에코 표현부터 다르다

WatchHand와 견주면 우리 재현은 거의 복제다. 신호(18–21 kHz, 48 kHz·600샘플), 칸 3.57 mm, 칸 300 정렬, 3줄 중앙값, 절댓값 차분, 2 × 60 × 96 입력, FastViT-T12, 손목 원점·손바닥 틀 정규화가 같다. 부호 있는 프로파일과 3줄 중앙값은 논문에 없고, 공개 자료를 열어 확인한 것이다 ([공개 자료 확인](public-watchhand-data.md)). 벗어나는 곳은 셋이다. 첫째, 처프에 삼각을 두었다. 계보에는 선례가 없다. 둘째, 워치8은 일부 경로에서 UNPROCESSED 대신 VOICE_RECOGNITION으로 녹음하고, 튐이 있다. 셋째, mm 잣대가 실측 손 길이가 아니라 MediaPipe가 추정한 손 크기 L(87–89 mm)이다. 공개 설정의 anchor_length(참가자 9는 73)로 보면 논문 mm는 우리 방식의 약 0.82배다 (잡음 바닥 손잡이(보고서 hand-sonar/2026-11-03-noise-floor-knobs)). 학습 쪽의 각도 칸 + 순기구학 머리, GRU + JEPA 순서 모델, 줄 단위 정규화는 우리가 계보보다 더 간 시험이다 ([연구 현황](research-status.md)).

계보 밖과의 차이는 더 크다. 아래 표는 차원별로 세 쪽을 나란히 놓는다.

| 차원 | 우리 | WatchHand와 Cornell 계보 | 계보 밖에서 흔한 선택 |
|---|---|---|---|
| 기기·배치 | 기성 Galaxy Watch8, 내장 1S+1M(맞은편 옆면), 앉음·팔 받침 없음 | WatchHand는 기성 워치 3종, 같은 배치, 팔꿈치를 책상에. 맞춤 기기는 S·M 두 쌍, 손바닥 쪽 쌍(EchoWrist), 기울인 보드(WristSonic, EchoForce) | 폰 마이크 2–7개(LLAP, Beyond-Voice), 귓속 마이크(ReHEarSSE, APG), 바깥 향 모듈(SonicASL), 덧붙인 마이크(EarHover, SonarWatch) |
| 신호 | C-FMCW, 톱니 또는 삼각, 창 없음 | 오름 처프만. 삼각·창은 보고 없음 | OFDM(FingerIO, ReHEarSSE), ZC·GSM 훈련열(VSkin, Strata, RobuCIR, SonicHand), CW 다중 톤(LLAP, SpiroSonic, APG), 순음 Doppler(SonicASL, MAF, EarHover), 백색잡음(BreathJunior), Hanning 창 FMCW(RAM-Hand, EchoPrint) |
| 대역·프레임 | 18–21 kHz, 48 kHz·600샘플(12.5 ms) | WatchHand는 같음. 맞춤은 50 kHz·600(12 ms), 이중 대역. 2025–26은 96–100 kHz로 9–10 kHz 폭 | 48 kHz에서 17–23 kHz 안 2–6 kHz 폭. 맞춤 외이도는 80–96 kHz로 16–48 kHz(EARFace, ECTF). APG는 30 kHz 넘음 |
| 처리·에코 표현 | 부호 있는 원형 상관, 칸 300, 3줄 중앙값, \|원본[t]\|−\|원본[t−1]\|, 원본 + 차분 | 같음. 계보 안에서도 차분 정의가 갈림(\|a\|−\|b\|, \|a−b\|, a−b) | 복소 CIR 차분, 칸 위상 추적, 정적 벡터 빼기, 전달함수·CFR, AR 계수, Doppler 스펙트로그램, 착용 직후 기준과의 차 |
| 정렬·동기 | 줄마다 직접 경로를 칸 300에 | 전체 정렬 + 미끄럼 창 봉우리 재정렬 + 중앙값(주기 밀림 대책) | 처음 한 번(VSkin, Beyond-Voice, RAM-Hand, EarCommand), 프레임마다 LOS(SonicHand), 실측 템플릿(EchoPrint), 위상으로 분수 지연(FingerIO, ReHEarSSE ≤4샘플), 시계 표류 선형 제거(MilliSonic, C-FMCW) |
| 모델·출력 | FastViT-T12(몸통은 바꿔도 됨), 60값 좌표. 각도 칸 + 순기구학, GRU + JEPA도 시험 | ResNet-18 → FastViT-T12, MSE + 0.1·속도 | CNN + LSTM(Beyond-Voice, RobuCIR), 관절 회전 + 골격 사전(SonicHand), TimeSformer(RAM-Hand), CTC(SonicASL, EarCommand, ReHEarSSE), 학습 없는 신호처리(FingerIO, LLAP, VSkin, 생리 대부분) |
| 정답 | 폰 정면 카메라 + MediaPipe(손목은 HaMeR), L = MediaPipe 추정 87–89 mm | MacBook 웹캠 + MediaPipe, 실측 손목–새끼 MCP | Leap Motion(SonicHand), OpenFace 2.0(EARFace), 얼굴 메시 교사(FaceListener), ARKit(EarIO), PSG·ECG·spirometer(생리) |
| 보정·적응 | 우리 자료로만 학습, 세션 나누기 = 착용 나누기, ±5칸 이동·마스킹 | 새 사용자를 뺀 기반 모델 + 미세조정. 자세 1세션 미세조정 9.96 → 7.65 mm | 착용 직후 기준 측정(EarCommand, ECTF), 착용마다 톤 고르기(APG), 정답 없는 가중(Commun Biol 2021), 2분 적응(EARFace), 적대적·대조 학습(SonicHand, RAM-Hand) |
| 견고성·실기기 | VOICE_RECOGNITION 대체, 줄 흔들림이 공개의 2배, 약 90샘플 튐, 볼륨 15 잡음, 삼각 바닥 | UNPROCESSED, 타이밍 지터·주기 밀림 보정. 대역 밖 소음은 무해, 걷기·손 모양 변화가 해 | 붙은 S·M 진폭 흔들림(ApneaApp), AGC·PAPR(ReHEarSSE), 트랜스듀서 응답 출렁임(MAF), 원시 마이크 접근 불가(EarCommand, EarSSR, SonarWatch), Android 10–20 ms 전달 간격(LLAP) |

표에서 우리 문제와 바로 닿는 차이는 세 줄이다. 계보 밖은 거의 모두 복소수(I/Q)에서 정적 성분을 빼고 위상을 쓴다. 착용형·근거리 연구는 착용마다 기준을 다시 잰다. 상용 기기 연구는 붙은 스피커·마이크의 진폭 흔들림과 오디오 경로를 따로 다룬다. 다음 절은 이것을 문제별로 옮긴다.

## 처리로 얻는 몫은 1–2 dB였고, 문헌이 크게 잰 손잡이는 배치와 신호 모으기다

먼저 우리 실측이 이미 닫은 길이다. 줄 공통 세기·위상 빼기는 오르내림을 삼각 +1.6, 톱니 +0.8, 한 창 +1.7 dB 올렸고 공개는 −0.8 dB였다. 맞춘 모형은 +4–5 dB를 예상했으니, 멈춤 변화의 대부분은 줄 공통이 아니다 (잡음 바닥 손잡이(보고서 hand-sonar/2026-11-03-noise-floor-knobs)). 반송파 위상을 버리는 포락선 차분은 오르내림을 삼각 −1.4, 톱니 −1.2, 공개 −1.0 dB 깎았다(같은 계산의 process.json(보고서 hand-sonar/2026-11-03-noise-floor-knobs)에서 묶음 평균을 냄). 줄 단위 정규화는 입력 상관을 공개 범위로 올렸지만 학습 비는 0.964에서 0.910으로 조금 나아졌고, 삼각 오름·내림 나누기 채널은 0.953–0.980으로 나아지지 않았다 ([연구 현황 5.9](research-status.md)). 위상·세기 변화 채널은 0.651에서 0.699로 나빠졌다 (위상 채널(보고서 hand-sonar/2026-11-01-phase-channels)). 볼륨은 11 → 13 → 15로 키워도 오르내림이 2.8 / 3.2 / 1.6 dB로 늘지 않았고, 대역 자리를 15–23.5 kHz로 옮긴 모의도 1 dB 안이었다 (잡음 바닥 손잡이(보고서 hand-sonar/2026-11-03-noise-floor-knobs)). 남는 차이는 움직이는 손가락의 반사 세기다.

근거 수준은 넷으로 적는다. **실측(우리)**은 우리 녹음에서 잰 것이다. **실측(문헌·가까운 조건)**은 기성 워치나 근거리 착용 기기에서 잰 것이다. **실측(문헌·다른 조건)**은 폰·스피커·맞춤 기기에서 잰 것이다. **추론**은 계산이나 논리뿐인 것이다.

### 문제 1: 반사 6 dB는 착용과 팔 자세를 먼저 바꿔 본다

우리 실측으로 공개와의 차이는 멈춤이 아니라 움직임에 있다. 톱니끼리 손 자리 멈춤은 −68.4 대 −69.1 dB로 같고, 움직임은 −63.2 대 −57.2 dB로 6 dB 약하다. 5–15 cm에서는 7.7 dB, 직접 경로 바로 옆은 1.7 dB 차이다. 정적 반사의 거리별 모양·크기도 같다 (잡음 바닥 손잡이(보고서 hand-sonar/2026-11-03-noise-floor-knobs)). 우리 손은 공개보다 2배 빠르고 1.5배 넓게 움직이는데도 그렇다 (우리 대 공개(보고서 hand-sonar/2026-11-03-ours-vs-public)). 손은 그 자리에 있는데, 움직일 때 돌아오는 변화가 약하다는 뜻이다. 처리로는 없는 신호를 만들 수 없다.

문헌에서 가장 크게 잰 효과는 배치였다. EchoSpeech는 스피커·마이크를 안경 다리에 두면 10자리 숫자 시험에서 50%가 안 됐고, 같은 아래 테에 두면 약 90%, 경로가 입술을 가로지르도록 반대편에 두면 98%였다 ([EchoSpeech](https://ruidongzhang.com/files/papers/EchoSpeech_authors_version.pdf)). EyeEcho는 자리에 따라 MAE가 18.0에서 22.8까지 갈렸고, WristSonic은 90° 기울인 배치가 가장 유익했다 ([EyeEcho](https://arxiv.org/html/2402.12388), [WristSonic](https://arxiv.org/html/2411.08217)). WatchHand는 EchoWrist가 손바닥 쪽 센서 덕에 더 강한 반사를 받았다고 보고, 그 차이로 4.81 대 7.87 mm를 들었다 ([WatchHand](https://arxiv.org/html/2602.21610)). 기성 워치는 스피커·마이크를 옮길 수 없다. 그러나 손과 워치의 기하는 바꿀 수 있다. WatchHand의 판 실험에서 판이 손의 긴축 방향(0°)으로 움직이면 무늬가 뚜렷했고, 긴축에서 60° 넘게 벗어나면 신호가 약했다 ([WatchHand](https://arxiv.org/html/2602.21610), 입력 피처 정리(보고서 hand-sonar/2026-11-01-input-features)). 손가락이 워치 시선 방향으로 움직이는 성분이 세기를 정한다는 뜻이다(추론). 우리 삼각 오름·내림 분석도 손가락의 워치 쪽 속도 성분이 작다고 봤다 ([연구 현황 5.9](research-status.md)).

WatchHand 쪽 근거는 기기보다 자세를 가리킨다. 세 워치의 10 cm 출력이 61.6–80.8 dBA로 달랐는데도, 재착용 MPJPE에 기기 효과가 유의하지 않았다(F 0.005–0.42, p 0.66–0.95) ([WatchHand](https://arxiv.org/html/2602.21610)). 참가자는 앉아서 팔꿈치를 책상에 올렸다(§4.3) (공개 대 우리 10-28(보고서 hand-sonar/2026-10-28-public-vs-ours)). 우리 세션은 모두 팔 받침이 없었다 (잡음 바닥 손잡이(보고서 hand-sonar/2026-11-03-noise-floor-knobs)). 우리가 바꿔 본 것은 손목 젖힘뿐이고, 0–45°에서 손가락 대비가 +2.4~+3.2 dB로 비슷했다 (입력 피처 정리(보고서 hand-sonar/2026-11-01-input-features)). 그래서 첫 시험은 팔 받침과 착용 위치·방향이다. 같은 동작을 1분씩 네 조건(지금 자세, 팔꿈치 책상, 워치를 손 쪽으로 1–2 cm, 워치를 손목 둘레로 조금 돌림)으로 찍고 5–15 cm 움직임 세기만 본다. "배치가 크게 중요하다"는 실측(문헌·다른 조건)이고, "우리 6 dB가 자세·착용 탓이다"는 추론이다.

둘째 손잡이는 주파수 조각 모으기다. BreathJunior는 6–21 kHz를 3 kHz 조각 다섯으로 나눠 조각마다 위상을 재고, 조각별 SNR로 가중해 더했다. 한 처프로 24 kHz를 다 쓰면 분해능이 1.4 cm지만 칸마다 SNR이 낮아, 분해능을 SNR과 바꾼 것이다 ([BreathJunior](https://homes.cs.washington.edu/~gshyam/Papers/whitenoise.pdf)). APG는 착용할 때마다 심박 대역 세기 비(PAR)가 가장 높은 톤을 골랐고, 최적 톤은 사람과 착용마다 바뀌었다 ([APG](https://storage.googleapis.com/gweb-research2023-media/pubtools/7284.pdf)). 스마트 스피커 심박 연구는 마이크 × 주파수 복소 가중을 정답 없이 처음 30 s로 최적화했다 ([Commun Biol 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7943557)). LoEar는 여러 부반송파를 코히어런트하게 더해 호흡 감지 거리를 2 m에서 7 m로 늘렸다 ([LoEar](https://doi.org/10.1145/3550293)). 우리 자료에도 같은 무늬가 있다. 18–21 kHz를 1 kHz씩 나누면 오르내림이 톱니 6.3 · 4.7 · 4.0 dB, 삼각 4.6 · 3.2 · 2.2 dB로 아래 조각이 가장 크고, 창 없는 17세션 모두 그랬다 (잡음 바닥 손잡이(보고서 hand-sonar/2026-11-03-noise-floor-knobs)). 우리 거리 축은 어차피 손가락 위치를 가리키지 않으므로(분해능 약 5.7 cm) (입력 피처 정리(보고서 hand-sonar/2026-11-01-input-features)), 분해능을 SNR과 바꾸는 대가가 작다(추론). 시험은 둘이다. 조각 셋을 따로 상관해 채널로 넣거나, 조각별 오르내림으로 가중해 한 채널로 합친다. 가중은 학습 세션 정답으로 정하거나 APG처럼 착용 직후 몇십 초로 정한다.

셋째 손잡이는 칸마다 정적 벡터를 빼고 위상을 읽는 것이다. LLAP은 톤마다 정적 벡터(직접 경로 + 멈춘 물체)를 천천히 갱신해 빼고, 남은 동적 벡터의 위상을 경로 길이로 바꿨다 ([LLAP](https://samsonsjarkal.github.io/KeSun/files/mobicom16_ultrasound.pdf)). SonarSelect는 이 구현을 Galaxy Watch 6에 옮기고 정적 벡터 추정을 0–10 cm에 맞춰 3.45–4.53 mm를 냈다 ([SonarSelect](https://arxiv.org/html/2503.02308v1)). SpiroSonic은 손에 든 폰의 흔들림으로 정적 반사가 바뀌자 I/Q 원호를 구간마다 맞춰 중심을 옮겼고, 5–20 cm에서 흉벽 오차가 3.8 mm 이하였다 ([SpiroSonic](https://par.nsf.gov/servlets/purl/10268386)). 우리 위상 채널은 이것과 다르다. 모든 칸이 함께 도는 공통 변화만 뺐고, 칸마다 멈춘 성분 S는 빼지 않았다. 우리 풀이도 방향을 읽으려면 칸마다 S를 빼야 한다고 적었다 (세기 차분과 위상(보고서 hand-sonar/2026-11-01-phase-vs-diff)). 그래서 "위상은 안 된다"는 결론은 아직 문헌 방식으로 시험되지 않았다. 근거는 실측(문헌·가까운 조건, 1D·한 반사체)이고, 손가락 여럿이 한 칸에 섞이는 우리 경우는 추론이다. 시험은 process_check에 "칸마다 느린 S를 빼고 복소 변화를 잰다"는 처리 하나를 더하는 것이다.

넷째는 긴 적분이다. ApneaApp은 처프 10개에 걸쳐 FFT를 해 칸 폭을 93.75 Hz에서 9.37 Hz로 줄였고, 스마트 스피커 심박은 50 ms 블록을 썼다 ([ApneaApp](https://homes.cs.washington.edu/~gshyam/Papers/apneaapp.pdf), [Commun Biol 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7943557)). 우리 차분은 늘 한 줄(12.5 ms) 간격이라, 느린 손가락 움직임이 한 줄에 남기는 변화가 작다. 차분 간격 k를 2·4·8로 늘린 오르내림은 저장된 프로파일로 바로 잰다. 줄 사이 잡음이 서로 독립이면 간격을 늘릴수록 오르내림이 오르고, 느린 흔들림이 섞여 있으면 오르지 않는다(추론).

### 문제 2: 줄 흔들림의 공통 몫은 이미 뺐고, 남은 것은 소리에 비례한다

직접 경로 세기의 줄 변화는 우리 0.30%, 공개 0.12%다 (잡음 바닥 풀이(보고서 hand-sonar/2026-11-03-noise-floor-explainer)). 그런데 앞에서 보았듯 줄 공통 몫을 빼도 1–2 dB뿐이다. 같은 착용에서 세기 40%와 80%로 찍은 녹음은 직접 경로 대비 바닥이 같았다. 잡음이 소리에 비례한다는 뜻이다. 볼륨 13에서 15로 올리면 직접 경로는 약 10 dB, 손 자리 바닥은 약 15 dB 올랐다 (신호 차이(보고서 hand-sonar/2026-10-31-signal-gap)). 정리하면 남은 멈춤 변화는 소리 자체가 만들고, 칸마다 다르다(실측(우리)).

문헌은 이런 잡음을 붙어 있는 스피커·마이크의 곱해지는 흔들림으로 다룬다. ApneaApp은 시험한 모든 폰에서 붙은 스피커·마이크로 동시에 내고 받으면 마이크 진폭이 예측할 수 없이 흔들린다고 보고했다. 진폭에 기대는 무호흡 판정에는 떨어진 마이크가 필요했고, 위상에 기대는 호흡수는 붙은 쌍으로도 됐다 ([ApneaApp](https://homes.cs.washington.edu/~gshyam/Papers/apneaapp.pdf)). FingerIO는 문턱을 직접 경로 진폭의 15%로 상대적으로 둬서, 비선형 진폭 요동이 오탐을 만들지 않게 했다 ([FingerIO](https://fingerio.cs.washington.edu/fingerio.pdf)). LibAS(MobiSys 2018, DOI 10.1145/3210240.3210312)는 기존 연구의 문제로 스피커 울림, 무작위 스피커 지터, AGC 적응을 꼽았다 ([LibAS](https://kabru.eecs.umich.edu/rtclweb/assets/publications/2018/paper_mobisys_final_acmart_upload.pdf)). EchoPrint는 하드웨어 때문에 설계와 달라진 실제 직접 경로를 조용한 곳에서 녹음해 템플릿으로 썼다 ([EchoPrint](https://par.nsf.gov/servlets/purl/10095814)). EarEcho는 처프로 스피커 응답을 재서 출력음을 추정하고 보상했다 ([EarEcho](https://dl.acm.org/doi/10.1145/3351239)). GazeTrak은 세기를 20%로 낮춰 운용했고 2%도 비슷했다 ([GazeTrak](https://arxiv.org/html/2402.14634)). 우리 40% = 80% 결과와 같은 방향이다.

옮길 것은 둘이다. 첫째, 줄 공통 보정은 더 다듬지 않는다. FingerIO식 분수 지연 보정이나 SonicHand식 프레임 정렬도 공통 성분을 더 정확히 빼는 방법이라, 남은 몫이 작다(실측(우리)). 둘째, 이상적 처프 대신 실제 수신 직접 경로를 템플릿으로 쓰거나, 문제 1의 셋째 손잡이처럼 칸마다 느린 정적 벡터를 따로 쫓는다. 손 없이 1분 녹음해 만든 템플릿으로 다시 상관하고, 지금 프로파일과 멈춤을 견주면 한 번에 갈린다. 이것이 워치의 칸별 잡음을 줄일지는 추론이다.

### 문제 3: 삼각 바닥은 나눗셈으로 지우되, 손 자리 몫은 0.4 dB다

삼각 바닥은 이론대로다. 오름과 내림이 같은 주파수를 두 번 지나며, 자기상관에 약 −20 dB의 평평한 바닥을 깐다. 모의도 실측과 같은 −22.3 dB를 냈다 (잡음 바닥 손잡이(보고서 hand-sonar/2026-11-03-noise-floor-knobs)). 계보에서 삼각을 쓴 논문은 없다. 오름·내림 처프를 쓴 SonarWatch는 거리 상관이 아니라 STFT 크기를 특징으로 썼다 ([SonarWatch](https://arxiv.org/abs/2408.12689)). 바닥을 다룬 선례도 없다.

삼각을 지키며 바닥을 지우는 길은 상관 대신 나눗셈이다. 원형 상관 프로파일 한 줄의 칸 방향 DFT는 Y·conj(X)다. 이를 \|X\|² + λ로 나누면 Y/X에 가까운 채널 추정이 되고, 삼각 스펙트럼의 물결이 상쇄된다(추론, 계산). 외이도 연구는 이미 이 꼴을 쓴다. EARFace는 CFR = FFT(수신)/FFT(송신), EarCommand는 수신/송신 비, ECTF는 역필터, RobuCIR와 FaceListener는 최소제곱 채널 추정을 쓴다 ([EARFace](https://www.cse.psu.edu/~mkg31/papers/earface.pdf), [EarCommand](https://dl.acm.org/doi/10.1145/3534613), [ECTF](https://dl.acm.org/doi/10.1145/3341163.3347747), [RobuCIR](https://shenjiaxing.win/pdf/wang_push_2020.pdf), [FaceListener](https://par.nsf.gov/servlets/purl/10335060)). λ는 꼭 필요하다. 우리 삼각 처프는 18.6 kHz에 −21 dB, 20.2 kHz에 −12 dB 골이 있어, 나누면 그 주파수의 잡음이 커진다 (잡음 바닥 손잡이(보고서 hand-sonar/2026-11-03-noise-floor-knobs)). 원본 녹음이 있으니 3줄 중앙값을 걸기 전 단계에서 시험할 수 있다. 다만 기대는 작게 잡는다. 같은 날 삼각과 톱니의 손 자리 오르내림은 4.9 대 5.3 dB로 0.4 dB 차이였다 (잡음 바닥 손잡이(보고서 hand-sonar/2026-11-03-noise-floor-knobs)). 오름·내림 나누기 채널도 학습을 돕지 않았다 ([연구 현황 5.9](research-status.md)). 나눗셈의 몫은 먼 칸 바닥과 30–40 cm 번짐을 걷어 내는 것이다.

소리를 다시 고를 일이 생기면, 딸깍도 바닥도 없는 신호가 있다. 18–21 kHz에서 크기가 평평하고 위상만 매끄러운 주기 신호를 주파수 쪽에서 합성하면, 처프가 되돌아가는 불연속이 없다. VSkin은 ZC를 주파수 쪽에서 보간해 썼고, ReHEarSSE는 같은 대역·길이에서 OFDM이 FMCW보다 나았다(89.7 대 82.5%, 외이도 과제). SonicHand는 ZC가 들리는 누설을 만들지 않는다고 했다 ([VSkin](https://samsonsjarkal.github.io/KeSun/files/mobicom18_vskin.pdf), [ReHEarSSE](https://dl.acm.org/doi/10.1145/3613904.3642095), [SonicHand](https://engineering.purdue.edu/~lusu/papers/TOSN2024.pdf)). 손 과제에서의 효과와 실제 들림은 확인되지 않았다(추론). 들리는지는 사용자 귀로만 판단할 수 있다. 지금 권하는 것은 삼각을 그대로 두는 나눗셈이다.

### 문제 4: 녹음 경로는 규격을 확인하고, 튄 줄은 버린다

VOICE_RECOGNITION 대체 자체는 규격과 우리 실측 모두 큰 문제가 아니라고 말한다. Android CDD 5.4.2는 이 음원이 기본으로 잡음 억제와 AGC를 꺼야 한다고 정한다(MUST). UNPROCESSED는 레벨 배율 말고는 처리가 없어야 한다 ([Android 15 CDD](https://source.android.com/docs/compatibility/15/android-15-cdd)). LibAS도 VOICE_RECOGNITION 플래그가 AGC를 끈다고 적었다 ([LibAS](https://kabru.eecs.umich.edu/rtclweb/assets/publications/2018/paper_mobisys_final_acmart_upload.pdf)). 우리도 같은 자리에서 두 음원을 1분씩 견줬고, 세기·18–21 kHz 대역·직접 경로·손 밖 바닥이 거의 같았다(−32.3 대 −32.4 dBFS) (공개 대 우리 10-28(보고서 hand-sonar/2026-10-28-public-vs-ours)). 남은 확인은 거칠게 한 번이면 된다. AudioManager.getProperty로 `PROPERTY_SUPPORT_MIC_NEAR_ULTRASOUND`, `PROPERTY_SUPPORT_SPEAKER_NEAR_ULTRASOUND`, `PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED`를 읽고, 녹음 중 입력에 붙은 효과(AEC·AGC·NS)를 본다. 음원별 기본 효과는 /vendor/etc/audio_effects.xml이 정한다 ([AOSP 전처리](https://source.android.com/docs/core/audio/implement-pre-processing)). CDD의 근초음파 규정은 18.5–20 kHz까지라 20–21 kHz는 규정 밖이다 ([Android 15 CDD](https://source.android.com/docs/compatibility/15/android-15-cdd)). 우리 20–21 kHz 조각이 가장 약한 것과 맞지만, 공개도 같은 경향(11.6 · 10.9 · 9.6 dB)이라 워치8만의 문제는 아니다.

튐은 문헌이 다루지 않았다. Android·Wear OS에서 스트림 중간에 수십 샘플이 뛰는 현상을 보고한 음향 감지 논문은 찾지 못했다. CDD 7.8.4의 끊김 없는 경로도 휴대기기 권고(SHOULD)일 뿐이다 ([Android 15 CDD](https://source.android.com/docs/compatibility/15/android-15-cdd)). 문헌의 정렬은 대부분 처음 한 번이다(VSkin, Beyond-Voice, RAM-Hand). 프레임마다 맞추는 것은 SonicHand(LOS 봉우리를 첫 칸으로 순환 이동)와 WatchHand(미끄럼 창 보정)다. WatchHand는 정렬이 어긋나면 프로파일이 틀린 거리로 밀려 예측이 완전히 틀린다고 썼다 ([SonicHand](https://engineering.purdue.edu/~lusu/papers/TOSN2024.pdf), [WatchHand](https://arxiv.org/html/2602.21610)). 우리는 이미 줄마다 직접 경로를 칸 300에 맞춘다. 남은 일은 튄 줄을 버리는 것이다. 원형 상관이라, 튐을 품은 줄은 두 위상이 섞인 줄이 된다(추론). 직접 경로가 앞 줄에서 정수 샘플만큼 옮긴 줄을 표시하고, 그 줄이 든 차분과 창을 학습·평가에서 뺀다. 지금 자료에는 급하지 않다. 11-03에 확인한 세션에는 끊기거나 되풀이된 줄이 없었다 (잡음 바닥 풀이(보고서 hand-sonar/2026-11-03-noise-floor-explainer)). 약 90샘플 튐은 폰–워치 거리 시험에서 측정의 약 4%에 나타났다 (실기기 정보(기기 시험 기록)).

### 한눈에 보는 옮길 거리

| 순 | 아이디어 | 겨냥 | 문헌 근거 | 근거 수준 | 우리 기록 | 첫 시험(거칠게) |
|---|---|---|---|---|---|---|
| 1 | 팔꿈치 책상, 워치 위치·방향 바꾸기 | 1 | EchoSpeech 배치 <50% → 98%, EyeEcho·WristSonic 자리 비교, WatchHand 판 실험·기기 효과 없음 | 실측(문헌·다른 조건과 가까운 조건). 우리 적용은 추론 | 손목 젖힘 0–45°는 대비 같음. 팔 받침·위치는 늘 같았음 | 같은 동작 1분씩 네 조건, 5–15 cm 움직임 세기 |
| 2 | 1 kHz 주파수 조각 가중·채널 | 1, 2 | BreathJunior 조각 SNR 가중, APG 착용마다 톤 고르기, Commun Biol 2021 복소 가중, LoEar 2 → 7 m | 실측(문헌·다른 조건) + 실측(우리: 아래 조각이 큼, 17/17) | 조각별 오르내림만 잼 | 조각 셋 채널 학습, 또는 가중 합의 오르내림 |
| 3 | 칸마다 느린 정적 벡터를 빼고 위상·복소 변화 | 1, 2 | LLAP LEVD, SonarSelect(Galaxy Watch 6) 3.45–4.53 mm, SpiroSonic 원호 ≤3.8 mm | 실측(문헌·가까운 조건, 1D) | 공통 회전만 뺀 위상 채널은 나빴음(0.651 → 0.699) | process_check에 처리 하나 추가 |
| 4 | 실측 수신 템플릿으로 상관 | 2 | EchoPrint 템플릿, EarEcho 스피커 응답 보상, ApneaApp 진폭 흔들림 | 실측(문헌·다른 조건). 워치는 추론 | 잡음이 소리에 비례(40% = 80%) | 손 없이 1분 → 템플릿, 멈춤 비교 |
| 5 | 차분 간격 늘리기 | 1 | ApneaApp 처프 10개 FFT, Commun Biol 2021 50 ms 블록 | 추론 | 차분은 늘 1줄 간격 | k = 2·4·8의 오르내림 |
| 6 | 상관 대신 나눗셈(Y/X, λ) | 3 | EARFace CFR, EarCommand 채널 응답, ECTF 역필터, RobuCIR·FaceListener 최소제곱 | 추론(계산) + 실측(우리: 삼각 골) | 같은 날 손 자리 차 0.4 dB, 나누기 채널 효과 없음 | 중앙값 전 프로파일의 칸 DFT ÷ (\|X\|² + λ) |
| 7 | 평평한 스펙트럼 주기 신호(ZC·OFDM) | 3 | VSkin ZC, ReHEarSSE OFDM 89.7 대 82.5%, SonicHand 누설 없음 주장 | 실측(문헌·다른 조건). 들림은 추론 | 없음 | 소리를 다시 고를 때만 |
| 8 | 음원 속성·입력 효과 확인 | 4 | CDD 5.4.2·5.11·7.8.3, AOSP 효과 설정, LibAS | 규격 + 실측(우리: 두 음원 같음) | 1분 비교 같음 | getProperty·효과 목록 한 번 |
| 9 | 튄 줄 표시·제외 | 4 | WatchHand(정렬 어긋남 → 완전히 틀린 예측), SonicHand 프레임 정렬 | 추론(스트림 중간 튐 보고 없음) | 확인한 세션엔 끊김·되풀이 줄 없음 | profiles 단계에 직접 경로 점프 검사 |

## 워치 오디오 경로·칸별 정적 변화·삼각 상관은 아무도 재지 않았다

문헌 쪽 빈칸은 세 종류다. 원문을 못 읽은 논문이 있고, 논문에 없는 세부가 있고, 아무도 재지 않은 현상이 있다. 마지막 종류는 우리 실측이 유일한 근거라는 뜻이다. 우리 쪽에는 이번에 제안한 시험이 모두 미시험으로 남아 있다.

| 무엇 | 상태 | 뜻 |
|---|---|---|
| Strata·CAT·UltraGesture·SoundWave·AudioGest 원문 | 초록·2차 자료만. UltraGesture는 신호 설명이 출처마다 다름(Barker 대 FHSS) | 고전 폰 추적의 프레임·정확도 일부 미확인 |
| SonarID의 워치 모델·대역·모델, SonarSelect의 대역·오디오 소스·코드(`RangeFinder.java`) | 초록·논문만 | Galaxy Watch에서 CW 위상이 어떤 음원·설정으로 돌았는지 모름 |
| C-FMCW·LoEar·DF-Sense·Zhang 2020·ACG·SonarBeat·BreathListener·BlinkListener·Second Chance·Sci Rep 2026·SpiroSense·EarEcho·EarDynamic의 신호 세부 | 초록만 | 부반송파 합성·SFO 보정 식 미확인 |
| [Experience](https://doi.org/10.1145/3495243.3560527)(DOI 10.1145/3495243.3560527)와 SonicHand 그림 5의 들리는 누설 수치 | 본문·그림 못 읽음 | ZC·OFDM이 삼각보다 덜 들리는지 근거 없음 |
| WatchHand 세부(60칸 시작점, 창 정규화 식, 미끄럼 창 크기, 중앙값 커널, 정답 프레임 위치, MediaPipe image/world, 속도 손실 식) | 논문에 없음 | 일부는 공개 자료로만 추정 |
| 워치7·8 스피커·마이크 부품과 응답, WatchHand의 워치8 시험 | 없음 | 6 dB가 기기 탓인지 문헌으로 못 가림 |
| Beyond-Voice·RAM-Hand의 정답 장치, EchoWrist 대역 불일치(20–24 대 18–21 kHz), WristSonic·SoniSpeech·SoniGAN 학회 | 노트에 없음·본문 불일치 | 방법 표 해당 칸 비움 |
| 삼각 처프 상관 바닥, 같은 자료의 \|a\|−\|b\| 대 \|a−b\|, 줄 이득·표본 이하 지터 정량, 스트림 중간 튐 | 어느 논문도 재지 않음 | 우리 실측이 유일한 근거 |
| Wear OS 오디오 경로의 지연·지터·UNPROCESSED 지원, 워치가 CDD 해당 절의 예외인지 | 공개 자료 없음 | 기기에서 직접 확인해야 함 |
| Cornell 밖 손목 연속 손 자세, 스마트워치 생리 능동 음향, Cornell 밖 안경형 얼굴·말 | 찾지 못함 | 손목 비교 대상은 WatchHand·SonarSelect·SonarID뿐 |
| 손 과제에서 OFDM·AR·전달함수 대 에코 프로파일, 손 자세용 대역·표본률 ablation | 없음 | 외이도의 이득이 손으로 옮겨질지 모름 |
| 우리 미시험: 팔 받침·착용 위치·방향, 주파수 조각 채널, 칸별 정적 벡터 빼기, 실측 템플릿, 나눗셈, 차분 간격, 튄 줄 검사 | 제안만 | 앞 절 표의 첫 시험으로 닫을 수 있음 |

## 결론

이번 비교로 무게중심이 처리에서 녹음으로 옮겨 간다. 우리 처리는 이미 WatchHand와 같고, 더 짤 몫은 1–2 dB뿐이다. 문헌이 크게 잰 효과는 둘 다 "어디를 볼지"를 바꾸는 일이었다. 맞춤 기기 연구는 스피커·마이크 자리로 정확도를 두 배 가까이 바꿨고, 생리 연구는 주파수·칸을 골라 모아 잡음 아래의 신호를 꺼냈다. 출력이 20 dB 가까이 다른 세 워치에서 기기 효과가 없었다는 WatchHand 결과까지 겹치면, 6 dB의 첫 용의자는 워치8 부품이 아니라 팔 받침과 착용이다(추론). 그래서 다음 걸음은 새 모델이 아니라 네 조건 1분 녹음과 주파수 조각 계산이고, 이것만으로 문제 1이 자세 쪽인지 기기 쪽인지 거칠게 갈린다.

두 음성 결과의 뜻도 좁혀 읽어야 한다. 위상 채널이 안 된 것은 "위상은 쓸모없다"가 아니라 "공통 회전만 뺀 위상은 보탬이 안 됐다"다. 문헌에서 위상이 mm를 만든 경우는 모두 칸이나 톤마다 정적 벡터를 먼저 뺐다. 오히려 반송파 위상을 버린 포락선이 오르내림을 깎았으니, 지금의 부호 있는 실수 차분이 위상 정보를 이미 일부 쓰고 있다는 뜻이다. 반대로 삼각 바닥은 그림에서 가장 눈에 띄지만 손 자리에서는 가장 작은 문제다. 더 멀리 보면, 이 분야 어디에나 있는 "사용자 독립은 무너지고 짧은 개인 보정으로 돌아온다"는 무늬는 세션마다 다시 차는 워치에도 그대로 온다. APG처럼 착용 직후 몇십 초로 이번 착용에서 손가락을 가장 잘 보는 주파수·칸을 고르는 단계는, 문제 1·2의 신호 대 잡음과 세션 나누기 일반화를 한 장치로 겨냥한다(추론).
