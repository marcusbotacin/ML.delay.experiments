for year in `seq 2012 1 2017`; do \
	for size in `seq 100 100 1000`; do \
		for drift in `seq 0 1 3`; do \
			for threshold in `seq 0.5 0.1 0.9`; do \
				for balance in `seq 0 1 10`; do \
					for partial in `seq 0 1 1`; do \
						for delay in `seq 0 1 50`; do \
							echo "python3 simulation_args2.py drebin_drift.parquet.zip" $year $size $drift $threshold $balance $partial $delay results; \
						done; \
					done; \
				done; \
			done; \
		done; \
	done; \
done;
