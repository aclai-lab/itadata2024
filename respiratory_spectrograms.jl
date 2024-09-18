using Audio911

sr = 8000
norm = true
nfft = 1024

# meditron anterior
wavfile = "/home/paso/Documents/Aclai/Datasets/health_recognition/databases/Respiratory_Sound_Database/audio_and_txt_files/101_1b1_Al_sc_Meditron.wav"
a1a = load_audio(source=wavfile, sr=sr, norm=norm)
a1 = get_linspec(source=get_stft(source=a1a, nfft=nfft), dbscale=true)
display(a1)

# litt3200 anterior
wavfile = "/home/paso/Documents/Aclai/Datasets/health_recognition/databases/Respiratory_Sound_Database/audio_and_txt_files/103_2b2_Ar_mc_LittC2SE.wav"
a2a = load_audio(source=wavfile, sr=sr, norm=norm)
a2 = get_linspec(source=get_stft(source=a2a, nfft=nfft), dbscale=true)
display(a2)

# littc2se anterior
wavfile = "/home/paso/Documents/Aclai/Datasets/health_recognition/databases/Respiratory_Sound_Database/audio_and_txt_files/104_1b1_Ar_sc_Litt3200.wav"
a3a = load_audio(source=wavfile, sr=sr, norm=norm)
a3 = get_linspec(source=get_stft(source=a3a, nfft=nfft), dbscale=true)
display(a3)

# akgc417 anterior
wavfile = "/home/paso/Documents/Aclai/Datasets/health_recognition/databases/Respiratory_Sound_Database/audio_and_txt_files/107_2b4_Al_mc_AKGC417L.wav"
a4a = load_audio(source=wavfile, sr=sr, norm=norm)
a4 = get_linspec(source=get_stft(source=a4a, nfft=nfft), dbscale=true)
display(a4)

# meditron posterior
wavfile = "/home/paso/Documents/Aclai/Datasets/health_recognition/databases/Respiratory_Sound_Database/audio_and_txt_files/110_1p1_Pr_sc_Meditron.wav"
a5a = load_audio(source=wavfile, sr=sr, norm=norm)
a5 = get_linspec(source=get_stft(source=a5a, nfft=nfft), dbscale=true)
display(a5)

# litt3200 posterior
wavfile = "/home/paso/Documents/Aclai/Datasets/health_recognition/databases/Respiratory_Sound_Database/audio_and_txt_files/112_1p1_Pl_sc_Litt3200.wav"
a6a = load_audio(source=wavfile, sr=sr, norm=norm)
a6 = get_linspec(source=get_stft(source=a6a, nfft=nfft), dbscale=true)
display(a6)

# litt c2se posterior
wavfile = "/home/paso/Documents/Aclai/Datasets/health_recognition/databases/Respiratory_Sound_Database/audio_and_txt_files/135_2b2_Pl_mc_LittC2SE.wav"
a7a = load_audio(source=wavfile, sr=sr, norm=norm)
a7 = get_linspec(source=get_stft(source=a7a, nfft=nfft), dbscale=true)
display(a7)

# akgc417 posterior
wavfile = "/home/paso/Documents/Aclai/Datasets/health_recognition/databases/Respiratory_Sound_Database/audio_and_txt_files/133_2p4_Pr_mc_AKGC417L.wav"
a8a = load_audio(source=wavfile, sr=sr, norm=norm)
a8 = get_linspec(source=get_stft(source=a8a, nfft=nfft), dbscale=true)
display(a8)