-- Keep a source file while allowing its transactions to be hidden from views.
ALTER TABLE public."CSVFile" ADD COLUMN IF NOT EXISTS is_hidden boolean NOT NULL DEFAULT false;
ALTER TABLE public."BillFile" ADD COLUMN IF NOT EXISTS is_hidden boolean NOT NULL DEFAULT false;
