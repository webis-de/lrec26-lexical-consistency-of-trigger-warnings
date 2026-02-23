ARG base_img

FROM $base_img
WORKDIR /app

# Get the jar
ARG jar_name
COPY ./ao3_tags/tags/${jar_name} /opt/apache/spark/jars

# Get the necessary files to run the job
COPY ./ao3_tags/conf /app/ao3_tags/conf
COPY ./ao3_tags/tags/requirements.txt /app/ao3_tags/tags/
COPY ./ao3_tags/resources /app/ao3_tags/resources
COPY ./ao3_tags/utils /app/ao3_tags/utils
COPY ./ao3_tags/__init__.py /app/ao3_tags/

ARG spark_version
RUN pip install -r "/app/ao3_tags/tags/requirements.txt"
RUN pip install pyspark==${spark_version}

# Add ao3_tags to path
ENV PYTHONPATH="/app/"