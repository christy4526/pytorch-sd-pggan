# SD-PGGAN (pytorch)

### 목표 : disentangled representation을 학습하면서 좋은 퀄리티의 고해상도 3D MRI를 생성하는 모델

SD-PGGAN : PG-GAN + SD-GAN(기존 PGGAN 훈련 방식에 SDGAN의 latent space decomposition 적용 )

![sd-pggan](https://user-images.githubusercontent.com/25657945/83388814-12403180-a42a-11ea-8300-a9ee6c881421.png)

### TODO
- CelebA-HQ data 생성 시 data가 가진 identity 정보를 포함하고 PGGAN이 사용할 수 있도록 하기
- SD-GAN 방식으로 PGGAN 훈련
- SD-PGGAN이 disentagled representation을 학습하는지 확인
