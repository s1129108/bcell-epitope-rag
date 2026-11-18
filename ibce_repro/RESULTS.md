# Tai tao iBCE-EL — ket qua

Cau hinh: `EM1 = (P_ERT + P_GB)/2`, ERT tren AAC+PCP (29 dim), GB tren DPC+PCP (409 dim).
Hyperparameter tu tune bang 5-fold CV tren train (Table S1 cua bai bao khong co trong
full text). Grid mo rong hai lan vi lan dau chon dung cau hinh o bien.

Cau hinh cuoi:
 - ERT: n_estimators=1000, max_features=0.8, min_samples_leaf=1
 - GB : n_estimators=500, learning_rate=0.05, max_depth=7

## CV tren benchmarking (10 lan x 5 fold)

| model | MCC | Acc | Sens | Spec | AUC | thr |
|---|---|---|---|---|---|---|
| ERT | 0.4174 | 0.7081 | 0.7249 | 0.6944 | 0.7614 | 0.450 |
| GB | 0.4081 | 0.7046 | 0.7050 | 0.7042 | 0.7591 | 0.439 |
| **iBCE-EL (EM1)** | **0.4301** | **0.7142** | 0.7336 | 0.6986 | **0.7724** | 0.434 |

std giua 10 lan rat nho (AUC 0.0017 cho EM1), nen cac so nay on dinh.

**Co che ensemble TAI LAP DUOC.** EM1 hon ERT +0.0110 AUC va hon GB +0.0133 AUC,
trong khi std chi 0.0017 — tuc gain that, khong phai nhieu. Day la ket qua dang chu y
nhat cua lan tai tao: viec trung binh hai probability cua hai model tren HAI FEATURE SET
KHAC NHAU cho gain on dinh ~0.011 AUC, gan dung muc bai bao bao cao (+0.006 den +0.009).

## Independent test (10 seed)

| model | nguong | MCC | Acc | Sens | Spec | AUC |
|---|---|---|---|---|---|---|
| ERT | 0.43 (CV) | 0.4150 | 0.7016 | 0.7664 | 0.6506 | 0.7655 |
| GB | 0.43 (CV) | 0.4129 | 0.7064 | 0.7199 | 0.6957 | 0.7574 |
| **iBCE-EL (EM1)** | **0.43 (CV)** | **0.4369** | **0.7161** | 0.7523 | 0.6876 | **0.7728** |
| iBCE-EL (EM1) | 0.35 (bai bao) | 0.4109 | 0.6876 | 0.8285 | 0.5766 | 0.7728 |

**Nguong 0.35 cua bai bao KHONG dung cho model cua ta.** Ap dung 0.35 cho ra
Sens 0.829 / Spec 0.577 — lech han, vi 0.35 la nguong hieu chuan cho phan bo probability
cua CHINH model ho. Nguong 0.43 chon tren OOF cua train cho profile gan ho hon nhieu
(Sens 0.752 / Spec 0.688 so voi 0.742 / 0.724 cua ho). Bai hoc: khi tai tao mot method
co nguong cong bo san, phai chon lai nguong tren train chu khong bê nguyen con so.

## Doi chieu voi so cong bo

| | ta (nguong CV) | bai bao | delta |
|---|---|---|---|
| **AUC (independent)** | **0.7728** | **0.789** | **−0.0162** |
| MCC | 0.4369 | 0.463 | −0.0261 |
| Acc | 0.7161 | 0.732 | −0.0159 |
| Sens | 0.7523 | 0.742 | +0.0103 |
| Spec | 0.6876 | 0.724 | −0.0364 |
| AUC (CV) | 0.7724 | 0.782 | −0.0096 |

**Ket luan: tai tao duoc ve co che va gan ve con so, thieu 0.010 (CV) den 0.016 (ind) AUC.**
Dung huong, dung cau truc, dung do lon gain cua ensemble, nhung khong cham duoc dinh.

Ba nguyen nhan co the, khong phan biet duoc bang du lieu hien co:
1. **Hyperparameter.** Table S1 nam o supplementary. Ho tune bang 5-fold CV LAP 10 LAN
   roi lay median hyperparameter; ta tune 1 lan de tiet kiem. Grid cua ta cung hep.
2. **Chi tiet feature chua ro trong bai.** Chuan hoa DPC (ta dung n−1), thang do cua
   PCP molecular weight va aliphatic index (ta khong normalize; tree bat bien voi
   scale don dieu nen it kha nang la nguyen nhan chinh).
3. **Chinh so 0.789 cua ho co loi the.** Sau ensemble model cho CV gan nhu nhau, va ho
   CHON EM1 bang cach xem performance tren independent set ("we checked its
   transferability on an independent data set"). Nghia la 0.789 da duoc chon qua
   independent set, trong khi 0.7728 cua ta thi khong. Phan nao cua khoang 0.016 la do nay
   thi khong the biet, nhung no di dung chieu.

## Y nghia voi muc tieu cua project

Khi so sanh TREN CUNG MOT LAN CAI DAT (AUC la metric khong phu thuoc nguong nen so sanh
duoc truc tiep), khoang cach thuc te nho hon nhieu so voi khi so voi so cong bo:

| | independent AUC |
|---|---|
| iBCE-EL tai tao (day) | **0.7728** |
| ESM C + MCNN + LSTM head (tot nhat cua project) | 0.7604 |
| ProtT5 + MCNN (fusion_cl, cnn_only) | 0.7293 |
| iBCE-EL **cong bo** | 0.789 |

Khoang cach con **0.0124 AUC**, khong phai 0.0286 nhu khi do voi so cong bo. Nghia la
muc tieu vuot iBCE-EL gan hon nhieu so vo truoc day van ghi. Nhung phai noi ro: so
0.789 van la so se bi reviewer dung de so sanh, nen "vuot 0.7728" khong phai la
"vuot iBCE-EL" trong bai bao.
