FROM algfw-sdk-agx_nx_aarch64:1.0

COPY mot/mot /usr/local/bin/mot

EXPOSE 18082
RUN ldconfig

WORKDIR /workspace

#add startup
CMD ["/usr/local/bin/mot"]
