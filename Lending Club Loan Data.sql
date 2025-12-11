/*

Cleaning Lending Club Loan Data in SQL Queries

*/


SELECT*
FROM [Portfolio Project].dbo.LendingClub_loan_data

--- Counting the number of rows to check if all the data was imported 
SELECT Count(*)
FROM dbo.LendingClub_loan_data

---Finding out if we have any empty columns
DECLARE @SQL NVARCHAR(MAX)='';

SELECT @SQL = @SQL + 
'SELECT ''' + COLUMN_NAME + ''' AS ColumnName, COUNT(['+ COLUMN_NAME +']) AS NotNullCount 
FROM LendingClub_loan_data UNION ALL '
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME='LendingClub_loan_data';

SET @SQL = LEFT(@SQL, LEN(@SQL)-10);

EXEC (@SQL);
--------From the above code we see that we have 17 columns that are empty in our dataset.



---Removing unwanted/unused columns
DECLARE @cols TABLE (col VARCHAR(200));

INSERT INTO @cols VALUES
('F1'),
('funded_amnt_inv'),
('emp_title'),
('pymnt_plan'),
('url'),
('desc'),
('title'),
('initial_list_status'),
('out_prncp'),
('out_prncp_inv'),
('total_pymnt'),
('total_pymnt_inv'),
('total_rec_prncp'),
('total_rec_int'),
('total_rec_late_fee'),
('recoveries'),
('collection_recovery_fee'),
('last_pymnt_d'),
('last_pymnt_amnt'),
('next_pymnt_d'),
('last_credit_pull_d'),
('policy_code'),
('annual_inc_joint'),
('dti_joint'),
('verification_status_joint'),
('open_acc_6m'),
('open_il_6m'),
('open_il_12m'),
('open_il_24m'),
('mths_since_rcnt_il'),
('total_bal_il'),
('il_util'),
('open_rv_12m'),
('open_rv_24m'),
('max_bal_bc'),
('all_util'),
('inq_fi'),
('total_cu_tl'),
('inq_last_12m');

-- Loop through each column and drop if exists
DECLARE @col VARCHAR(200), @sql NVARCHAR(300);

SELECT @col = MIN(col) FROM @cols;

WHILE @col IS NOT NULL
BEGIN
    IF EXISTS (
        SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = 'LendingClub_loan_data'
          AND COLUMN_NAME = @col
    )
    BEGIN
        SET @sql = N'ALTER TABLE LendingClub_loan_data DROP COLUMN ' + QUOTENAME(@col);
        EXEC(@sql);
    END

    SELECT @col = MIN(col) FROM @cols WHERE col > @col;
END;


---Checking the formats of the columns
EXEC sp_help 'LendingClub_loan_data';


--- Formatting the date columns to 'DATE' format
ALTER TABLE LendingClub_loan_data ALTER COLUMN issue_d DATE;

UPDATE LendingClub_loan_data
SET issue_d = CONVERT(date, issue_d);

ALTER TABLE LendingClub_loan_data ALTER COLUMN earliest_cr_line DATE;

UPDATE LendingClub_loan_data
SET earliest_cr_line = CONVERT(date, earliest_cr_line);

---Formatting all the amount columns to DECIMAL
ALTER TABLE LendingClub_loan_data ALTER COLUMN loan_amnt DECIMAL(18,2);
ALTER TABLE LendingClub_loan_data ALTER COLUMN funded_amnt DECIMAL(18,2);
ALTER TABLE LendingClub_loan_data ALTER COLUMN int_rate DECIMAL(5,2);
ALTER TABLE LendingClub_loan_data ALTER COLUMN installment DECIMAL(18,2);
ALTER TABLE LendingClub_loan_data ALTER COLUMN annual_inc DECIMAL(18,2);
ALTER TABLE LendingClub_loan_data ALTER COLUMN dti DECIMAL(18,2);
ALTER TABLE LendingClub_loan_data ALTER COLUMN revol_bal DECIMAL(18,2);
ALTER TABLE LendingClub_loan_data ALTER COLUMN revol_util DECIMAL(5,2);
ALTER TABLE LendingClub_loan_data ALTER COLUMN tot_coll_amt DECIMAL(18,2);
ALTER TABLE LendingClub_loan_data ALTER COLUMN tot_cur_bal DECIMAL(18,2);
ALTER TABLE LendingClub_loan_data ALTER COLUMN total_rev_hi_lim DECIMAL(18,2);

--- Formatting all the count columns to INT
ALTER TABLE LendingClub_loan_data ALTER COLUMN delinq_2yrs INT;
ALTER TABLE LendingClub_loan_data ALTER COLUMN inq_last_6mths INT;
ALTER TABLE LendingClub_loan_data ALTER COLUMN mths_since_last_delinq INT;
ALTER TABLE LendingClub_loan_data ALTER COLUMN mths_since_last_record INT;
ALTER TABLE LendingClub_loan_data ALTER COLUMN open_acc INT;
ALTER TABLE LendingClub_loan_data ALTER COLUMN pub_rec INT;
ALTER TABLE LendingClub_loan_data ALTER COLUMN total_acc INT;
ALTER TABLE LendingClub_loan_data ALTER COLUMN collections_12_mths_ex_med INT;
ALTER TABLE LendingClub_loan_data ALTER COLUMN mths_since_last_major_derog INT;
ALTER TABLE LendingClub_loan_data ALTER COLUMN acc_now_delinq INT;

---Formatting the identifier columns to BIGINT
ALTER TABLE LendingClub_loan_data ALTER COLUMN id BIGINT;
ALTER TABLE LendingClub_loan_data ALTER COLUMN member_id BIGINT;

---Checking for Duplicates
SELECT 
id, COUNT(*) as duplicate_id
FROM LendingClub_loan_data
GROUP BY id
HAVING COUNT(*) > 1;

-----The above resulted in 0 duplicate loan id's.

---Finding out if missing values exist in the dataset
DECLARE @SQL NVARCHAR(MAX) = '';

SELECT @SQL = @SQL +
'SELECT ''' + COLUMN_NAME + ''' AS ColumnName,
        COUNT(*) AS TotalRows,
        SUM(CASE 
                WHEN [' + COLUMN_NAME + '] IS NULL THEN 1
                WHEN TRY_CAST([' + COLUMN_NAME + '] AS VARCHAR(255)) = '''' THEN 1
                ELSE 0
            END) AS MissingValues
 FROM LendingClub_loan_data
 UNION ALL '
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'LendingClub_loan_data';

-- Remove last UNION ALL
SET @SQL = LEFT(@SQL, LEN(@SQL) - 10);

EXEC(@SQL);




	



