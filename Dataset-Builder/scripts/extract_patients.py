import sqlalchemy
import pandas as pd

# Database connection
engine = sqlalchemy.create_engine('postgresql://username:password@localhost:5432/mimic')

# Query for CHF patients with BNP > 50
query_chf = """
SELECT DISTINCT p.subject_id
FROM diagnoses_icd d
JOIN patients p ON d.subject_id = p.subject_id
JOIN labevents l ON p.subject_id = l.subject_id
WHERE d.icd9_code LIKE '428%'
  AND l.itemid = 50963
  AND l.valuenum > 50
"""
chf_patients = pd.read_sql_query(query_chf, engine)


# Query for healthy patients (no CHF diagnosis)
query_healthy = """
SELECT DISTINCT p.subject_id
FROM patients p
WHERE p.subject_id NOT IN (
    SELECT subject_id
    FROM diagnoses_icd
    WHERE icd9_code LIKE '428%'
)
"""
healthy_patients = pd.read_sql_query(query_healthy, engine)
