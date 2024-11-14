import streamlit as st
import redis
import pickle
import time
import matplotlib.pyplot as plt


MEAN = -35.80143737792969
STD = 16.67781639099121
AUDIO_Y_LIMITS = (-32768, 32767) # Adjust as needed


def display_prediction(placeholder, score, label):
    color = "green" if label == "Normal" else "red"
    placeholder.markdown(
        f"<div style='background-color:{color}; color:white; text-align:center; padding:10px; font-size:20px;'>"
        f"{label} with score {score:.3f}</div>", unsafe_allow_html=True)


def display_raw_audio(placeholder, audio):
    fig, ax = plt.subplots()
    ax.plot(audio, color='blue')
    ax.set_ylim(AUDIO_Y_LIMITS)
    placeholder.pyplot(fig)
    plt.close(fig)


def display_features(placeholder, features):
    mel_spec_db = (features * STD) + MEAN
    fig, ax = plt.subplots()
    cax = ax.imshow(mel_spec_db, aspect='auto', origin='lower')
    fig.colorbar(cax)
    placeholder.pyplot(fig)
    plt.close(fig)


def main():
    r = redis.Redis(host='localhost', port=6379, db=0)
    st.title('Real-Time Audio Analysis')

    col1, col2 = st.columns(2)
    audio_plot_placeholder = col1.empty()
    feature_plot_placeholder = col2.empty()
    
    prediction_placeholder = st.empty()
    
    while True:
        keys_mel = sorted(r.keys('mel_spec:*'))
        keys_raw = sorted(r.keys('raw_audio:*'))
        if keys_mel:
            latest_key_raw = keys_raw[-1]
            timestamp_raw = latest_key_raw.decode('utf-8').split(':')[1]
            raw_audio = pickle.loads(r.get(latest_key_raw))

            latest_key_mel = keys_mel[-1]
            timestamp_mel = latest_key_mel.decode('utf-8').split(':')[1]
            mel_spec_db = pickle.loads(r.get(latest_key_mel))
            prediction_score = r.get(f'prediction:{timestamp_mel}')

            if prediction_score:
                prediction_score = float(prediction_score)
                label = 'Abnormal' if prediction_score > 0.5 else 'Normal'

                # Update audio plot
                display_raw_audio(audio_plot_placeholder, raw_audio)

                # Update spectrogram
                display_features(feature_plot_placeholder, mel_spec_db)

                # Update prediction text
                display_prediction(prediction_placeholder, prediction_score, label=label)

        time.sleep(0.1)
   
if __name__ == '__main__':
    main()
