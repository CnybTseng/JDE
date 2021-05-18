FROM algfw-sdk-agx_nx_aarch64:1.0

COPY mot /usr/local/bin/mot

# Port
EXPOSE 18082
RUN ldconfig

WORKDIR /workspace
COPY jde.trt /workspace
COPY lib* /workspace/

# Execute when docker runing
CMD ["/usr/local/bin/mot"]