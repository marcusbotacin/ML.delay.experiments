for i in `ls *.exp*`; do echo -n $i" "; cat $i | cut -d',' -f257; done
