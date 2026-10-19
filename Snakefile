rule download_test_data:
    output: "data/raw/immrep_test_set.csv"
    run:
        import kagglehub
        from pathlib import Path

        download_path = kagglehub.competition_download('immrep26')
        download_path = Path(download_path)
        (download_path / 'test.csv').move(Path(output[0]))

