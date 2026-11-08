-- =====================================================================
-- Credit Risk Prediction: SQL Workflow
-- ID/X Partners x Rakamin Academy
--
-- Dialect : PostgreSQL. Bagian lain juga jalan di SQLite, kecuali STEP 1b.
-- Urutan  : jalankan dari STEP 1 sampai STEP 14 secara berurutan.
-- HASIL   : angka dari data asli, bisa dipakai untuk mencocokkan hasilmu.
-- =====================================================================


-- =====================================================================
-- STEP 1: CREATE THE RAW TABLE
-- Membuat tabel kosong dengan 75 kolom, urutannya sama dengan CSV.
--
-- HASIL:
--   Tidak ada tabel hasil. Tabel loan_raw terbentuk dengan 75 kolom, masih kosong.
-- PENJELASAN:
--   Tipe kolom disiapkan: angka sebagai NUMERIC, teks sebagai TEXT.
--   Kolom pertama CSV tanpa nama, jadi di sini kuberi nama row_index.
-- =====================================================================
DROP VIEW IF EXISTS loan_features;
DROP VIEW IF EXISTS loan_labeled;
DROP TABLE IF EXISTS loan_raw;

CREATE TABLE loan_raw (
    row_index                   INTEGER,   -- kolom pertama CSV, tanpa nama
    id                          BIGINT,
    member_id                   BIGINT,
    loan_amnt                   NUMERIC,
    funded_amnt                 NUMERIC,
    funded_amnt_inv             NUMERIC,
    term                        TEXT,
    int_rate                    NUMERIC,
    installment                 NUMERIC,
    grade                       TEXT,
    sub_grade                   TEXT,
    emp_title                   TEXT,
    emp_length                  TEXT,
    home_ownership              TEXT,
    annual_inc                  NUMERIC,
    verification_status         TEXT,
    issue_d                     TEXT,
    loan_status                 TEXT,
    pymnt_plan                  TEXT,
    url                         TEXT,
    "desc"                      TEXT,
    purpose                     TEXT,
    title                       TEXT,
    zip_code                    TEXT,
    addr_state                  TEXT,
    dti                         NUMERIC,
    delinq_2yrs                 NUMERIC,
    earliest_cr_line            TEXT,
    inq_last_6mths              NUMERIC,
    mths_since_last_delinq      NUMERIC,
    mths_since_last_record      NUMERIC,
    open_acc                    NUMERIC,
    pub_rec                     NUMERIC,
    revol_bal                   NUMERIC,
    revol_util                  NUMERIC,
    total_acc                   NUMERIC,
    initial_list_status         TEXT,
    out_prncp                   NUMERIC,
    out_prncp_inv               NUMERIC,
    total_pymnt                 NUMERIC,
    total_pymnt_inv             NUMERIC,
    total_rec_prncp             NUMERIC,
    total_rec_int               NUMERIC,
    total_rec_late_fee          NUMERIC,
    recoveries                  NUMERIC,
    collection_recovery_fee     NUMERIC,
    last_pymnt_d                TEXT,
    last_pymnt_amnt             NUMERIC,
    next_pymnt_d                TEXT,
    last_credit_pull_d          TEXT,
    collections_12_mths_ex_med  NUMERIC,
    mths_since_last_major_derog NUMERIC,
    policy_code                 NUMERIC,
    application_type            TEXT,
    annual_inc_joint            NUMERIC,
    dti_joint                   NUMERIC,
    verification_status_joint   TEXT,
    acc_now_delinq              NUMERIC,
    tot_coll_amt                NUMERIC,
    tot_cur_bal                 NUMERIC,
    open_acc_6m                 NUMERIC,
    open_il_6m                  NUMERIC,
    open_il_12m                 NUMERIC,
    open_il_24m                 NUMERIC,
    mths_since_rcnt_il          NUMERIC,
    total_bal_il                NUMERIC,
    il_util                     NUMERIC,
    open_rv_12m                 NUMERIC,
    open_rv_24m                 NUMERIC,
    max_bal_bc                  NUMERIC,
    all_util                    NUMERIC,
    total_rev_hi_lim            NUMERIC,
    inq_fi                      NUMERIC,
    total_cu_tl                 NUMERIC,
    inq_last_12m                NUMERIC
);


-- =====================================================================
-- STEP 1b: IMPORT THE CSV
-- Ganti path sesuai lokasi file. Baris ini perintah psql, khusus PostgreSQL.
-- Kalau pakai pgAdmin atau DBeaver, boleh pakai menu Import.
--
-- HASIL:
--   Tidak ada tabel hasil. Tabel loan_raw terisi 466.285 baris.
-- PENJELASAN:
--   Kalau jumlah baris di STEP 2 cocok, berarti impor berhasil.
-- =====================================================================
\copy loan_raw FROM 'loan_data_2007_2014.csv' WITH (FORMAT csv, HEADER true);


-- =====================================================================
-- STEP 2: CHECK DATASET SIZE AND DUPLICATES
--
-- HASIL:
--   466.285 baris, 466.285 id unik, 466.285 member_id unik, 0 id ganda.
-- PENJELASAN:
--   Tidak ada pinjaman tercatat dua kali, jadi tidak perlu menghapus duplikat.
--   Setiap peminjam muncul sekali, sehingga tidak ada riwayat berulang per peminjam.
-- =====================================================================
SELECT
    COUNT(*)                         AS total_rows,
    COUNT(DISTINCT id)               AS unique_loan_id,
    COUNT(DISTINCT member_id)        AS unique_member_id,
    COUNT(*) - COUNT(DISTINCT id)    AS duplicate_loan_id
FROM loan_raw;


-- =====================================================================
-- STEP 3: CHECK MISSING VALUES
-- Melihat kolom penting yang kosong. COUNT(kolom) tidak menghitung NULL.
--
-- HASIL:
--   annual_inc_joint 100%, mths_since_last_record 86,57%, mths_since_last_delinq 53,69%,
--   tot_cur_bal 15,07%, emp_length 4,51%, revol_util 0,07%, annual_inc mendekati 0%.
-- PENJELASAN:
--   annual_inc_joint kosong total, jadi tidak berguna dan dibuang.
--   Dua kolom mths_since kosong bukan karena data hilang. Artinya belum pernah ada kejadian,
--   jadi dibuat flag had_delinq dan had_public_record di STEP 13.
--   tot_cur_bal kosong terutama pada pinjaman lama. emp_length kosong diberi flag sendiri.
-- =====================================================================
SELECT 'emp_length' AS column_name,
       ROUND(100.0 * (COUNT(*) - COUNT(emp_length)) / COUNT(*), 2) AS missing_pct FROM loan_raw
UNION ALL SELECT 'annual_inc',
       ROUND(100.0 * (COUNT(*) - COUNT(annual_inc)) / COUNT(*), 2) FROM loan_raw
UNION ALL SELECT 'revol_util',
       ROUND(100.0 * (COUNT(*) - COUNT(revol_util)) / COUNT(*), 2) FROM loan_raw
UNION ALL SELECT 'mths_since_last_delinq',
       ROUND(100.0 * (COUNT(*) - COUNT(mths_since_last_delinq)) / COUNT(*), 2) FROM loan_raw
UNION ALL SELECT 'mths_since_last_record',
       ROUND(100.0 * (COUNT(*) - COUNT(mths_since_last_record)) / COUNT(*), 2) FROM loan_raw
UNION ALL SELECT 'tot_cur_bal',
       ROUND(100.0 * (COUNT(*) - COUNT(tot_cur_bal)) / COUNT(*), 2) FROM loan_raw
UNION ALL SELECT 'annual_inc_joint',
       ROUND(100.0 * (COUNT(*) - COUNT(annual_inc_joint)) / COUNT(*), 2) FROM loan_raw
ORDER BY missing_pct DESC;


-- =====================================================================
-- STEP 4: EXPLORE LOAN STATUS
-- loan_status adalah kolom performa kredit, dasar penentuan label.
--
-- HASIL:
--   Current 224.226 (48,09%), Fully Paid 184.739 (39,62%), Charged Off 42.475 (9,11%),
--   Late 31-120 hari 6.900 (1,48%), In Grace Period 3.146 (0,67%), Not meet policy Fully Paid 1.988 (0,43%),
--   Late 16-30 hari 1.218 (0,26%), Default 832 (0,18%), Not meet policy Charged Off 761 (0,16%).
-- PENJELASAN:
--   Hampir separuh pinjaman masih berjalan, jadi hasil akhirnya belum diketahui.
--   Karena itu tidak semua baris bisa dipakai untuk belajar.
-- =====================================================================
SELECT
    loan_status,
    COUNT(*)                                                 AS total_loans,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)       AS pct_of_all
FROM loan_raw
GROUP BY loan_status
ORDER BY total_loans DESC;


-- =====================================================================
-- STEP 5: CREATE THE GOOD/BAD LABEL
-- target = 1 untuk BAD, 0 untuk GOOD.
-- Status yang hasilnya belum jelas tidak ikut (Current, In Grace Period,
-- Late 16-30 days). Tahun dan bulan cair juga dipisah dari issue_d ('Dec-11').
--
-- HASIL:
--   Tidak ada tabel hasil. View loan_labeled terbentuk, berisi 6 status yang hasilnya sudah jelas.
-- PENJELASAN:
--   GOOD berarti lunas. BAD berarti gagal bayar atau menunggak lebih dari 30 hari.
--   Late 16-30 hari dan In Grace Period dikeluarkan karena masih bisa kembali lancar.
-- =====================================================================
CREATE VIEW loan_labeled AS
SELECT
    r.*,
    CASE
        WHEN loan_status IN ('Charged Off', 'Default', 'Late (31-120 days)',
                             'Does not meet the credit policy. Status:Charged Off') THEN 1
        ELSE 0
    END AS target,
    2000 + CAST(SUBSTR(issue_d, 5, 2) AS INTEGER) AS issue_year,
    CASE SUBSTR(issue_d, 1, 3)
        WHEN 'Jan' THEN 1 WHEN 'Feb' THEN 2 WHEN 'Mar' THEN 3 WHEN 'Apr' THEN 4
        WHEN 'May' THEN 5 WHEN 'Jun' THEN 6 WHEN 'Jul' THEN 7 WHEN 'Aug' THEN 8
        WHEN 'Sep' THEN 9 WHEN 'Oct' THEN 10 WHEN 'Nov' THEN 11 WHEN 'Dec' THEN 12
    END AS issue_month
FROM loan_raw r
WHERE loan_status IN ('Fully Paid', 'Charged Off', 'Default', 'Late (31-120 days)',
                      'Does not meet the credit policy. Status:Fully Paid',
                      'Does not meet the credit policy. Status:Charged Off');


-- =====================================================================
-- STEP 6: CALCULATE THE OVERALL BAD RATE
--
-- HASIL:
--   237.695 pinjaman berlabel, 50.968 BAD, 186.727 GOOD, bad rate 21,44%.
-- PENJELASAN:
--   Sekitar 1 dari 5 pinjaman berakhir buruk. Kelas BAD lebih sedikit (imbalanced), jadi akurasi
--   saja tidak cukup untuk menilai model.
-- =====================================================================
SELECT
    COUNT(*)                                AS labeled_loans,
    SUM(target)                             AS bad_loans,
    COUNT(*) - SUM(target)                  AS good_loans,
    ROUND(100.0 * SUM(target) / COUNT(*), 2) AS bad_rate_pct
FROM loan_labeled;


-- =====================================================================
-- STEP 7: CALCULATE THE BUSINESS COST OF GOOD AND BAD LOANS
-- Arus kas bersih = total pembayaran - dana cair. Dasar angka 3,7 kali di deck.
-- Belum termasuk biaya dana dan biaya operasional.
--
-- HASIL:
--   GOOD rata-rata untung $2.038, BAD rata-rata rugi $7.599.
-- PENJELASAN:
--   7.599 dibagi 2.038 = 3,7. Satu kredit buruk menghapus keuntungan sekitar 3,7 kredit baik.
--   Sebagian pinjaman BAD masih menunggak, jadi kerugian akhirnya bisa berubah.
-- =====================================================================
SELECT
    CASE target WHEN 1 THEN 'BAD' ELSE 'GOOD' END  AS label,
    COUNT(*)                                       AS total_loans,
    ROUND(AVG(total_pymnt - funded_amnt), 0)       AS avg_net_cash_per_loan
FROM loan_labeled
GROUP BY target;


-- =====================================================================
-- STEP 8: ANALYZE BAD RATE BY ISSUE YEAR
-- Memperlihatkan bias waktu: bad rate 2013-2014 tampak tinggi.
--
-- HASIL:
--   2007 26,20%, 2008 20,73%, 2009 13,69%, 2010 14,04%, 2011 16,52%, 2012 16,73%, 2013 22,95%, 2014 26,22%.
-- PENJELASAN:
--   Lonjakan 2013-2014 belum tentu berarti peminjamnya memburuk. STEP 9 menjelaskan sebabnya.
--   Tahun 2007 tinggi, tetapi datanya hanya 603 pinjaman.
-- =====================================================================
SELECT
    issue_year,
    COUNT(*)                                 AS labeled_loans,
    ROUND(100.0 * SUM(target) / COUNT(*), 2) AS bad_rate_pct
FROM loan_labeled
GROUP BY issue_year
ORDER BY issue_year;


-- =====================================================================
-- STEP 9: CHECK THE SHARE OF LOANS STILL RUNNING
-- Penyebab bias di STEP 8: 67,3% pinjaman 2014 masih Current.
--
-- HASIL:
--   Porsi Current: 2011 9,0%, 2012 6,5%, 2013 44,7%, 2014 67,3%, tahun sebelumnya hampir 0%.
-- PENJELASAN:
--   Sebagian besar pinjaman 2014 belum selesai dan dikeluarkan dari label. Yang tersisa cenderung
--   pinjaman yang cepat selesai, termasuk yang cepat gagal, sehingga bad rate terlihat tinggi.
-- =====================================================================
SELECT
    2000 + CAST(SUBSTR(issue_d, 5, 2) AS INTEGER)                                      AS issue_year,
    COUNT(*)                                                                           AS all_loans,
    ROUND(100.0 * SUM(CASE WHEN loan_status = 'Current' THEN 1 ELSE 0 END) / COUNT(*), 1) AS current_pct
FROM loan_raw
GROUP BY 2000 + CAST(SUBSTR(issue_d, 5, 2) AS INTEGER)
ORDER BY issue_year;


-- =====================================================================
-- STEP 10: CALCULATE THE MATURED COHORT BAD RATE
-- Tenor 36 bulan yang cair sampai 2012 sudah terpantau penuh.
--
-- HASIL:
--   74.959 pinjaman, bad rate 13,09%.
-- PENJELASAN:
--   Ini gambaran yang lebih adil karena semua pinjamannya sudah terpantau penuh.
--   Bandingkan dengan 26,22% pada 2014 di STEP 8. Jadi angka 2014 jangan dibaca sebagai tingkat gagal bayar sebenarnya.
-- =====================================================================
SELECT
    COUNT(*)                                 AS matured_loans,
    ROUND(100.0 * SUM(target) / COUNT(*), 2) AS bad_rate_pct
FROM loan_labeled
WHERE TRIM(term) = '36 months'
  AND issue_year <= 2012;


-- =====================================================================
-- STEP 11: ANALYZE BAD RATE BY BORROWER AND LOAN SEGMENT
-- Mencari segmen paling berisiko. Satu query untuk setiap variabel.
-- =====================================================================
-- 11a. Grade
-- HASIL: A 7,34%, B 14,95%, C 23,70%, D 30,83%, E 38,88%, F 43,89%, G 47,02%.
-- PENJELASAN: Variabel paling membedakan. Bad rate naik teratur dari grade terbaik ke terburuk.
SELECT grade, COUNT(*) AS loans, ROUND(100.0 * SUM(target) / COUNT(*), 2) AS bad_rate_pct
FROM loan_labeled GROUP BY grade ORDER BY grade;

-- 11b. Term
-- HASIL: 60 bulan 34,92%, 36 bulan 17,67%.
-- PENJELASAN: Tenor panjang lebih berisiko, tetapi sebagian dipengaruhi bias waktu dari STEP 9.
SELECT TRIM(term) AS term, COUNT(*) AS loans, ROUND(100.0 * SUM(target) / COUNT(*), 2) AS bad_rate_pct
FROM loan_labeled GROUP BY TRIM(term) ORDER BY bad_rate_pct DESC;

-- 11c. Purpose
-- HASIL: Tertinggi small_business 31,75%, terendah car 13,75% dan wedding 14,13%.
-- PENJELASAN: debt_consolidation paling banyak (138.318 pinjaman) dengan bad rate 22,46%.
SELECT purpose, COUNT(*) AS loans, ROUND(100.0 * SUM(target) / COUNT(*), 2) AS bad_rate_pct
FROM loan_labeled GROUP BY purpose ORDER BY bad_rate_pct DESC;

-- 11d. Verification status
-- HASIL: Verified 24,23%, Source Verified 23,45%, Not Verified 16,85%.
-- PENJELASAN: Terbalik dari dugaan. Kemungkinan verifikasi lebih sering diminta untuk pengajuan yang sudah dinilai berisiko.
SELECT verification_status, COUNT(*) AS loans, ROUND(100.0 * SUM(target) / COUNT(*), 2) AS bad_rate_pct
FROM loan_labeled GROUP BY verification_status ORDER BY bad_rate_pct DESC;

-- 11e. Home ownership
-- HASIL: RENT 23,60%, OWN 22,58%, MORTGAGE 19,39%. NONE dan ANY sangat sedikit (49 pinjaman).
-- PENJELASAN: Karena terlalu sedikit, NONE dan ANY digabung ke OTHER di STEP 13.
SELECT home_ownership, COUNT(*) AS loans, ROUND(100.0 * SUM(target) / COUNT(*), 2) AS bad_rate_pct
FROM loan_labeled GROUP BY home_ownership ORDER BY bad_rate_pct DESC;

-- 11f. Employment length
-- HASIL: Semua kategori 20,04-22,44%, tetapi yang kosong (NULL) 30,43%.
-- PENJELASAN: Lama bekerja hampir tidak membedakan. Justru data yang kosong lebih berisiko, jadi diberi flag emp_length_missing.
SELECT emp_length, COUNT(*) AS loans, ROUND(100.0 * SUM(target) / COUNT(*), 2) AS bad_rate_pct
FROM loan_labeled GROUP BY emp_length ORDER BY bad_rate_pct DESC;


-- =====================================================================
-- STEP 12: CHECK OUTLIERS IN ANNUAL INCOME
-- NTILE(100) membagi data jadi 100 kelompok. Kelompok 100 = 1% tertinggi.
-- Hanya memakai data train supaya data test tidak ikut memengaruhi.
-- Pembatasan nilai (winsorizing) tetap dilakukan di Python.
--
-- HASIL:
--   Kelompok 1: $1.896-$17.500. Kelompok 50: $60.000. Kelompok 99: $185.000-$230.000.
--   Kelompok 100: $230.000-$7.141.778.
-- PENJELASAN:
--   Penghasilan sangat miring ke kanan: nilai tengahnya $60.000, tertingginya lebih dari $7 juta.
--   Nilai ekstrem bisa saja benar, jadi barisnya tidak dihapus, melainkan dibatasi lewat winsorizing di Python.
-- =====================================================================
WITH income_rank AS (
    SELECT annual_inc, NTILE(100) OVER (ORDER BY annual_inc) AS pct_group
    FROM loan_labeled
    WHERE issue_year <= 2013 AND annual_inc IS NOT NULL
)
SELECT
    pct_group,
    MIN(annual_inc) AS min_income,
    MAX(annual_inc) AS max_income,
    COUNT(*)        AS loans
FROM income_rank
WHERE pct_group IN (1, 50, 99, 100)
GROUP BY pct_group
ORDER BY pct_group;


-- =====================================================================
-- STEP 13: BUILD THE MODELING DATASET
-- Hanya informasi yang tersedia saat pengajuan. Kolom pasca-pencairan seperti
-- total_pymnt, recoveries, dan last_pymnt_d sengaja tidak dipilih (data leakage).
-- data_split: train sampai 2013, test sepanjang 2014 (out-of-time).
--
-- HASIL:
--   Tidak ada tabel hasil. View loan_features terbentuk, 237.695 baris dan 34 kolom.
-- PENJELASAN:
--   Kolom pasca-pencairan tidak ikut. Kalau ikut, AUC bisa mencapai 0,999 tetapi palsu.
--   Kolom tot_coll_amt, tot_cur_bal, dan total_rev_hi_lim masih kosong sekitar 28%, dan diisi median di Python.
-- =====================================================================
CREATE VIEW loan_features AS
WITH base AS (
    SELECT
        l.*,
        2000 + CAST(SUBSTR(earliest_cr_line, 5, 2) AS INTEGER) AS cr_year_raw,
        CASE SUBSTR(earliest_cr_line, 1, 3)
            WHEN 'Jan' THEN 1 WHEN 'Feb' THEN 2 WHEN 'Mar' THEN 3 WHEN 'Apr' THEN 4
            WHEN 'May' THEN 5 WHEN 'Jun' THEN 6 WHEN 'Jul' THEN 7 WHEN 'Aug' THEN 8
            WHEN 'Sep' THEN 9 WHEN 'Oct' THEN 10 WHEN 'Nov' THEN 11 WHEN 'Dec' THEN 12
        END AS cr_month
    FROM loan_labeled l
),
months AS (
    SELECT
        b.*,
        -- Tahun dua digit: '68' terbaca 2068, jadi dikurangi 100 tahun bila setelah tanggal cair.
        CASE
            WHEN cr_year_raw * 12 + cr_month > issue_year * 12 + issue_month
                THEN (issue_year * 12 + issue_month) - ((cr_year_raw - 100) * 12 + cr_month)
            ELSE (issue_year * 12 + issue_month) - (cr_year_raw * 12 + cr_month)
        END AS credit_history_months
    FROM base b
)
SELECT
    id,
    target,
    issue_year,
    CASE WHEN issue_year <= 2013 THEN 'train' ELSE 'test' END AS data_split,

    -- Loan
    loan_amnt,
    CASE WHEN TRIM(term) = '60 months' THEN 1 ELSE 0 END AS term_60,
    purpose,
    (CASE grade WHEN 'A' THEN 0 WHEN 'B' THEN 5 WHEN 'C' THEN 10 WHEN 'D' THEN 15
                WHEN 'E' THEN 20 WHEN 'F' THEN 25 WHEN 'G' THEN 30 END)
        + CAST(SUBSTR(sub_grade, 2, 1) AS INTEGER) AS sub_grade_num,

    -- Borrower
    annual_inc,
    CASE emp_length
        WHEN '< 1 year' THEN 0 WHEN '1 year' THEN 1 WHEN '2 years' THEN 2 WHEN '3 years' THEN 3
        WHEN '4 years' THEN 4 WHEN '5 years' THEN 5 WHEN '6 years' THEN 6 WHEN '7 years' THEN 7
        WHEN '8 years' THEN 8 WHEN '9 years' THEN 9 WHEN '10+ years' THEN 10
    END AS emp_years,
    CASE WHEN emp_length IS NULL THEN 1 ELSE 0 END AS emp_length_missing,
    CASE WHEN home_ownership IN ('NONE', 'ANY') THEN 'OTHER' ELSE home_ownership END AS home_ownership,
    verification_status,

    -- Credit history
    dti,
    revol_util,
    revol_bal,
    delinq_2yrs,
    inq_last_6mths,
    open_acc,
    pub_rec,
    total_acc,
    collections_12_mths_ex_med,
    acc_now_delinq,
    tot_coll_amt,
    tot_cur_bal,
    total_rev_hi_lim,
    ROUND(1.0 * credit_history_months / 12, 2) AS credit_history_years,
    mths_since_last_delinq,
    mths_since_last_record,
    CASE WHEN mths_since_last_delinq IS NOT NULL THEN 1 ELSE 0 END AS had_delinq,
    CASE WHEN mths_since_last_record IS NOT NULL THEN 1 ELSE 0 END AS had_public_record,

    -- Ratios
    CASE WHEN annual_inc > 0 THEN 1.0 * loan_amnt / annual_inc END        AS loan_to_income,
    CASE WHEN annual_inc > 0 THEN 12.0 * installment / annual_inc END     AS installment_to_income,
    CASE WHEN annual_inc > 0 THEN 1.0 * revol_bal / annual_inc END        AS revol_bal_to_income
FROM months;


-- =====================================================================
-- STEP 14: VALIDATE THE TRAIN AND TEST SPLIT
-- Setelah cocok, dataset ini siap dipakai di Python:
--   SELECT * FROM loan_features;
--
-- HASIL:
--   Train 163.541 pinjaman, 31.527 BAD, bad rate 19,28%, tahun 2007-2013.
--   Test 74.154 pinjaman, 19.441 BAD, bad rate 26,22%, tahun 2014.
-- PENJELASAN:
--   Model belajar dari masa lalu dan diuji pada pinjaman yang lebih baru, sama seperti pemakaian nyata.
--   Bad rate test lebih tinggi karena bias waktu dari STEP 9, bukan karena pembagiannya salah.
-- =====================================================================
SELECT
    data_split,
    COUNT(*)                                 AS loans,
    SUM(target)                              AS bad_loans,
    ROUND(100.0 * SUM(target) / COUNT(*), 2) AS bad_rate_pct,
    MIN(issue_year)                          AS first_year,
    MAX(issue_year)                          AS last_year
FROM loan_features
GROUP BY data_split
ORDER BY data_split DESC;
