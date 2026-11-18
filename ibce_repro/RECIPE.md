# iBCE-EL — cong thuc goc, trich tu bai bao

Manavalan B, Govindaraj RG, Shin TH, Kim MO, Lee G.
"iBCE-EL: A New Ensemble Learning Framework for Improved Linear B-Cell Epitope Prediction."
Front Immunol. 2018;9:1695. doi:10.3389/fimmu.2018.01695. PMC6072840.
Full text lay tu Europe PMC REST (`.../rest/PMC6072840/fullTextXML`).

## 1. Dataset — KHOP CHINH XAC voi `dataset/` cua chung ta

IEDB, chi giu peptide 5-25 residue. Positive = duoc xac nhan positive trong >=2 thi
nghiem B-cell doc lap. Negative = negative trong >=2 thi nghiem va CHUA TUNG positive
trong bat ky assay nao (7 assay: qualitative binding, decreased disease, neutralization,
KD, ADCC, off rate, on rate). CD-HIT loai bo cap co identity > 70%.
=> nr set 5550 BCE + 6893 non-BCE, chia random 80/20:

| | BCE | non-BCE | tong |
|---|---|---|---|
| benchmarking (train) | 4440 | 5485 | 9925 |
| independent | 1110 | 1408 | 2518 |

Doi chieu `dataset/*.txt` cua chung ta: 4440 / 5485 / 1110 / 1408 — **khop tung so**.

[MOD] Hai diem lech can ghi nho, khong phai loi cua ta:
 - Bai bao noi giu 5-25 residue. File cua ta co min = 11, tuc ban tai ve tu web server
   da khac ban dung trong bai (hoac bai bao ghi nham nguong duoi).
 - `Ind-negative.txt` co **34 chuoi dai > 25** (toi da 49). Moi pipeline dat MAXSEQ=25
   deu dang am tham cat 34 chuoi nay (1.4% independent negative). Voi iBCE-EL thi khong
   anh huong vi feature cua no la composition do dai bat ky.

## 2. Feature construction — 5 composition + 6 binary profile

| ten | dim | cach tinh |
|---|---|---|
| **AAC** | 20 | ty le % cua 20 amino acid chuan |
| **DPC** | 400 | ty le % cua 400 dipeptide |
| **CTD** | 147 | Composition-Transition-Distribution (Dubchak 1995). 20 aa chia 3 nhom (polar / neutral / hydrophobic) theo tung PCP. C = 3 gia tri, T = 3 gia tri, D = 5 gia tri moi nhom = 15 => **21 feature moi PCP**. 7 PCP (hydrophobicity, polarizability, normalized van der Waals volume, secondary structure, polarity, charge, solvent accessibility) => 7 x 21 = 147 |
| **AAI** | 20 | 8 amino acid index "chat luong cao" tu AAindex, chon bang fuzzy clustering cua Saha 2012: **BLAM930101, BIOV880101, MAXF760101, TSAJ990101, NAKH920108, CEDJ970104, LIFS790101, MIYS990104**. Ban 160-dim = 8 index x 20 aa. Bai bao noi hai ban cho ket qua tuong duong nen **dung ban 20-dim = trung binh 8 index** |
| **PCP** | 9 | (i) ty le residue hydrophobic (F,I,W,L,V,M,Y,C,A); (ii) hydrophilic (S,Q,T,R,K,N,D,E); (iii) neutral (H,G,P); (iv) positively charged (K,H,R); (v) negatively charged (D,E); (vi) turn-forming = (N+G+P+S)/n; (vii) absolute charge per residue; (viii) molecular weight; (ix) aliphatic index = (A + 2.9V + 3.9I + 3.9L)/n |
| **BPF** | 100-400 | one-hot 20-dim moi residue, chi lay dau/cuoi chuoi. 6 bien the: BPFN5 (100), BPFC5 (100), BPFN5+BPFC5 (200), BPFN10 (200), BPFC10 (200), BPFN10+BPFC10 (400) |

**17 hybrid** (to hop cua 5 composition):
H1 AAC+AAI | H2 AAC+DPC+AAI | H3 AAC+DPC+AAI+CTD | H4 AAC+DPC+AAI+CTD+PCP |
H5 AAC+DPC | H6 AAC+CTD | H7 **AAC+PCP** | H8 AAI+DPC | H9 AAI+DPC+CTD |
H10 AAI+DPC+CTD+PCP | H11 AAI+CTD | H12 **AAI+PCP** | H13 DPC+CTD | H14 DPC+CTD+PCP |
H15 **DPC+PCP** | H16 CTD+DPC | H17 AAC+AAI+PCP

Khong co feature selection nao. Khong co PLM, khong co embedding, khong co deep learning.

## 3. Classification model

6 thuat toan: SVM, RF, ERT (extremely randomized trees), GB (gradient boosting),
AB (AdaBoost), k-NN. Moi thuat toan x moi feature set => 27 classifier duoc danh gia.
Hyperparameter toi uu bang 5-fold CV tren benchmarking, **lap lai 10 lan** voi partition
random khac nhau, lay **median** cua hyperparameter va **average** cua performance.

Feature tot nhat tim duoc cho tung thuat toan:
 - **ERT -> H7 = AAC + PCP (29 dim)**
 - **GB  -> H15 = DPC + PCP (409 dim)**
 - RF -> H12 = AAI + PCP (29 dim)
 - AB -> PCP (9 dim)
 - SVM, k-NN -> BPFN10+BPFC10 (400 dim)

Ket luan cua ho: decision-tree based (RF/ERT/GB) tot hon han SVM/AB/kNN bat ke feature.

**Ensemble:** 6 to hop duoc thu — EM1 (GB+ERT), EM2 (+RF), EM3 (+SVM), EM4 (+AB),
EM5 (+kNN), EM6 (GB+SVM+ERT). Cach ghep la **trung binh don gian cua predicted
probability**: `P_EM = (1/n) * sum_i P_i`. Khong co stacking, khong co meta-classifier,
khong co trong so.

**iBCE-EL = EM1 = average(P_GB on DPC+PCP, P_ERT on AAC+PCP)**, nguong 0.35.

Nguong duoc chon bang grid search toi da hoa MCC. Nguong toi uu cua tung model:
iBCE-EL 0.35, RF 0.47, ERT 0.45, GB 0.26, AB 0.50, SVM 0.41, kNN 0.41.

[MOD] Mot diem ve phuong phap cua chinh bai bao can noi ro: 6 ensemble model cho
performance gan nhu nhu nhau tren CV, nen ho **dung independent set de chon EM1**
("we checked its transferability on an independent data set and selected a model that
showed consistent performance both on benchmarking and independent data sets").
Nghia la independent test cua iBCE-EL **khong hoan toan sach** — no da tham gia vao
buoc chon model. Khi tai tao, ta chon EM1 theo bai bao (khong lap lai buoc ro ri nay),
va khi so sanh phuong phap cua ta voi iBCE-EL thi nen nho rang 0.789 cua ho co loi the nay.

## 4. Protocol danh gia

5-fold CV tren benchmarking (lap 10 lan, lay average) + independent test.
Metric: Sens, Spec, Acc, MCC, AUC. McNemar chi-square de so sanh.

## 5. So can tai tao

| | MCC | Acc | Sens | Spec | AUC |
|---|---|---|---|---|---|
| CV tren benchmarking | 0.454 | 0.729 | 0.716 | 0.739 | 0.782 |
| **independent test** | **0.463** | **0.732** | **0.742** | **0.724** | **0.789** |

Cac baseline trong bai (independent): GB 0.776 AUC, ERT 0.780, RF 0.777, AB 0.742,
LBtope 0.781, kNN 0.685, SVM 0.694.

## 6. Diem con thieu thong tin, phai tu quyet dinh khi cai dat

1. **AAI 20-dim tinh the nao chua ro.** "Trung binh cua 8 index" cho ra 1 gia tri moi
   amino acid (20 gia tri), nhung de thanh feature cua MOT PEPTIDE thi phai ket hop voi
   composition. Cach hop ly nhat: `AAI[i] = AAC[i] * mean_8_index(aa_i)`, tuc composition
   co trong so bang index. Se thu ca bien the "chi mean index tren cac residue co mat".
2. **Chuan hoa index.** AAindex co thang do rat khac nhau giua cac index; bai bao khong
   noi co normalize truoc khi lay trung binh. Se normalize z-score tren 20 amino acid.
3. **Nhom 3 lop cua CTD cho 7 PCP** la bang tra cuu chuan cua Dubchak/iFeature, bai bao
   dan sang bai truoc. Se dung dung bang cua iFeature.
4. **Hyperparameter cu the** nam o Table S1 (supplementary), khong co trong full text.
   Se tu tune bang 5-fold CV tren train, dung grid tuong tu.
