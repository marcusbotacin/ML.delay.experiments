FROM python:3.7
WORKDIR /ml
RUN pip install --upgrade pip
RUN pip install "numpy<1.20"
RUN pip install pandas==1.3.4
RUN pip install pyarrow
RUN pip install scipy
RUN pip install scikit-learn
RUN pip install scikit-multiflow
RUN pip install gensim
RUN pip install pytz
RUN pip install python-dateutil
COPY *.py /ml/
COPY *.zip /ml/
