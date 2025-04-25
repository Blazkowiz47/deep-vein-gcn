FROM python:3.12.9
WORKDIR /root/code

# Personal Setup for developemnt inside the container
RUN curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-x86_64.appimage
RUN chmod u+x nvim-linux-x86_64.appimage
RUN ./nvim-linux-x86_64.appimage --appimage-extract
RUN ./squashfs-root/AppRun --version

RUN mv squashfs-root /
RUN ln -s /squashfs-root/AppRun /usr/bin/nvim

RUN apt update
RUN git clone https://github.com/Blazkowiz47/nvim-config.git /root/.config/nvim
RUN git clone https://github.com/github/copilot.vim.git ~/.config/nvim/pack/github/start/copilot.vim
RUN apt install luarocks npm ripgrep fd-find -y
RUN luarocks install jsregexp
RUN npm cache clean -f
RUN npm install -g n
RUN n stable
RUN npm install -g tree-sitter-cli
RUN ln -s /root/.config/nvim/.tmux.conf /root/.tmux.conf
RUN apt install python3-venv -y
RUN apt install tmux -y
RUN echo 'alias ta="tmux attach"' >> /root/.bashrc
ENV TERM=xterm-256color

# Dependencies
RUN pip install --upgrade pip --no-cache-dir
RUN pip install torch torchvision --no-cache-dir
RUN pip install fvcore --no-cache-dir
RUN pip install einops --no-cache-dir
RUN pip install timm --no-cache-dir
RUN pip install wandb --no-cache-dir
RUN pip install torchmetrics --no-cache-dir

# Keep container alive:
CMD ["/bin/bash"]

