FROM python:3.8

# Install the function's dependencies using file requirements.txt
# from your project folder.

COPY requirements.txt ./
RUN pip3.8 install -r requirements.txt

RUN mkdir /assets
COPY /assets/styles.css /assets/

COPY all_attributes_for_widget_minimal.bin ./

COPY app.py ./



#RUN apt-get update --fix-missing && apt-get install libxrender1




WORKDIR ./


# Copy function code
#should be split into various files to avoid uploading .bin each time
#COPY . ${LAMBDA_TASK_ROOT}
EXPOSE 8050

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
#CMD [ "app.handler" ]
CMD ["python", "./app.py"]