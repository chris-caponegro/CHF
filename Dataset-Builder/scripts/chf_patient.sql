SELECT DISTINCT p.subject_id, d.hadm_id
FROM diagnoses_icd d
JOIN patients p ON d.subject_id = p.subject_id
JOIN labevents l ON d.hadm_id = l.hadm_id
WHERE d.icd9_code LIKE '428%'
  AND l.itemid IN (225664, 227442)
  AND l.valuenum IS NOT NULL
  AND (
    (l.itemid = 225664 AND l.valuenum > 100)
    OR
    (l.itemid = 227442 AND l.valuenum > 300)
  );
