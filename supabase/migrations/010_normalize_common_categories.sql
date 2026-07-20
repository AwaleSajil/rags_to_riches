-- Consolidate existing common category variants into the labels used by the app.
UPDATE public."Transaction"
SET category = CASE lower(trim(category))
    WHEN 'grocery' THEN 'Groceries'
    WHEN 'groceries' THEN 'Groceries'
    WHEN 'supermarket' THEN 'Groceries'
    WHEN 'supermarkets' THEN 'Groceries'
    WHEN 'restaurant' THEN 'Dining'
    WHEN 'restaurants' THEN 'Dining'
    WHEN 'dining' THEN 'Dining'
    WHEN 'food & dining' THEN 'Dining'
    WHEN 'takeout' THEN 'Dining'
    WHEN 'transport' THEN 'Transportation'
    WHEN 'transportation' THEN 'Transportation'
    WHEN 'gas' THEN 'Transportation'
    WHEN 'fuel' THEN 'Transportation'
    WHEN 'rideshare' THEN 'Transportation'
    WHEN 'retail' THEN 'Shopping'
    WHEN 'shopping' THEN 'Shopping'
    WHEN 'utility' THEN 'Utilities'
    WHEN 'utilities' THEN 'Utilities'
    WHEN 'medical' THEN 'Healthcare'
    WHEN 'healthcare' THEN 'Healthcare'
    WHEN 'uncategorized' THEN 'Uncategorized'
    ELSE category
END
WHERE category IS NOT NULL;
