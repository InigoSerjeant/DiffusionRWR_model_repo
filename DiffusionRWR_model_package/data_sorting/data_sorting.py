from __future__ import annotations

import glob
import os
import warnings
from pathlib import Path

import pandas as pd


def load_data(folder_path: str | Path) -> dict[str, pd.DataFrame]:
	"""
	Load and row-standardize all CSV files from a modelled data folder.
	"""
	folder_path = str(folder_path)
	csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

	print(f"Number of CSV files found: {len(csv_files)}")
	if csv_files:
		print("\nCSV files found:")
		for file_path in csv_files:
			print(f"  - {os.path.basename(file_path)}")

	data_dict: dict[str, pd.DataFrame] = {}
	for file_path in csv_files:
		name = os.path.splitext(os.path.basename(file_path))[0]
		df = pd.read_csv(file_path)

		if len(df.columns) > 0 and (df.columns[0] == "Unnamed: 0" or "gene" in df.columns[0].lower()):
			df.set_index(df.columns[0], inplace=True)

		integer_cols = [col for col in df.columns if float(col) % 1 == 0]
		df = df[integer_cols]
		print(f"  Filtered to integer time points: {integer_cols}")

		df_standardized = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
		data_dict[name] = df_standardized
		print(f"Imported '{name}' with shape: {df.shape} (row-wise standardized)")

	print(f"\nSuccessfully imported {len(data_dict)} datasets.")
	print(f"Available datasets: {list(data_dict.keys())}")

	return data_dict


def load_and_process_modelled_data(
	folder_path: str | Path | None = None,
	histone_dataset_filter: str = "all",
	require_rna: bool = True,
) -> dict[str, pd.DataFrame]:
	"""
	Load datasets from `data/Modelled` and clean names exactly like diffusion functions.

	Naming behavior:
	- Any RNA dataset key -> ``rna_ai``
	- Histone filter set (e.g. ``k20me3``) -> only keep that histone as the cleaned name
	- Filter ``all`` -> keep recognized histones as ``k9me2``, ``k20me3``, ``k27me3``

	Parameters
	----------
	folder_path
		Folder containing modelled CSV files. If None, defaults to package data/Modelled.
	histone_dataset_filter
		Histone token to keep (e.g., ``k20me3``) or ``all``.
	require_rna
		If True, raise an error when no RNA dataset is found.
	"""
	if folder_path is None:
		folder_path = Path(__file__).resolve().parents[1] / "data" / "Modelled"

	data_dict = load_data(folder_path)
	selected_histone = str(histone_dataset_filter).strip().lower()

	cleaned_data_dict: dict[str, pd.DataFrame] = {}
	for key, df in data_dict.items():
		key_lower = key.lower()

		if "rna" in key_lower:
			clean_name = "rna_ai"
		elif selected_histone and selected_histone != "all":
			if selected_histone not in key_lower:
				continue
			clean_name = selected_histone
		else:
			if "k9me2" in key_lower:
				clean_name = "k9me2"
			elif "k20me3" in key_lower:
				clean_name = "k20me3"
			elif "k27me3" in key_lower:
				clean_name = "k27me3"
			else:
				clean_name = key

		cleaned_data_dict[clean_name] = df

	if require_rna and "rna_ai" not in cleaned_data_dict:
		raise ValueError("RNA dataset (rna_ai) was not found in the selected folder.")

	if selected_histone and selected_histone != "all":
		if not any("rna" not in name for name in cleaned_data_dict.keys()):
			raise ValueError(
				f"No histone dataset matched filter '{selected_histone}' in folder '{folder_path}'."
			)

	return cleaned_data_dict


DEFAULT_MODELLED_FOLDER = Path(__file__).resolve().parents[1] / "data" / "Modelled"


def _build_std_data_dict() -> dict[str, pd.DataFrame]:
	try:
		return load_and_process_modelled_data(
			folder_path=DEFAULT_MODELLED_FOLDER,
			histone_dataset_filter="all",
			require_rna=True,
		)
	except Exception as exc:
		warnings.warn(
			f"Could not build std_data_dict from '{DEFAULT_MODELLED_FOLDER}': {exc}",
			RuntimeWarning,
		)
		return {}


std_data_dict = _build_std_data_dict()

