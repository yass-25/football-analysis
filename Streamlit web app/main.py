import tempfile
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
from ultralytics import YOLO
from detection import create_colors_info, detect

def main():

    st.set_page_config(page_title="Application d'analyse tactique de football avec IA", layout="wide", initial_sidebar_state="expanded")
    st.title("Analyse Tactique de Football par IA")
    st.subheader(":red[Ne fonctionne qu'avec une caméra tactique]")

    st.sidebar.title("Paramètres Principaux")
    demo_selected = st.sidebar.radio(label="Sélectionnez une vidéo de démonstration", options=["Démo 1", "Démo 2"], horizontal=True)

    st.sidebar.markdown('---')
    st.sidebar.subheader("Téléversement de Vidéo")
    input_vide_file = st.sidebar.file_uploader('Importer un fichier vidéo', type=['mp4','mov', 'avi', 'm4v', 'asf'])

    demo_vid_paths={
        "Démo 1":'./demo_vid_1.mp4',
        "Démo 2":'./demo_vid_2.mp4'
    }
    demo_vid_path = demo_vid_paths[demo_selected]
    demo_team_info = {
        "Démo 1":{"team1_name":"France",
                  "team2_name":"Suisse",
                  "team1_p_color":'#1E2530',
                  "team1_gk_color":'#F5FD15',
                  "team2_p_color":'#FBFCFA',
                  "team2_gk_color":'#B1FCC4',
                  },
        "Démo 2":{"team1_name":"Chelsea",
                  "team2_name":"Manchester City",
                  "team1_p_color":'#29478A',
                  "team1_gk_color":'#DC6258',
                  "team2_p_color":'#90C8FF',
                  "team2_gk_color":'#BCC703',
                  }
    }
    selected_team_info = demo_team_info[demo_selected]

    tempf = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    if not input_vide_file:
        tempf.name = demo_vid_path
        demo_vid = open(tempf.name, 'rb')
        demo_bytes = demo_vid.read()

        st.sidebar.text('Vidéo de démonstration')
        st.sidebar.video(demo_bytes)
    else:
        tempf.write(input_vide_file.read())
        demo_vid = open(tempf.name, 'rb')
        demo_bytes = demo_vid.read()

        st.sidebar.text('Vidéo importée')
        st.sidebar.video(demo_bytes)

    model_players = YOLO("../models/Yolo8L Players/weights/best.pt")
    model_keypoints = YOLO("../models/Yolo8M Field Keypoints/weights/best.pt")
    tracker = DeepSort(max_age=30)

    st.sidebar.markdown('---')
    st.sidebar.subheader("Noms des Équipes")
    team1_name = st.sidebar.text_input(label='Nom de la première équipe', value=selected_team_info["team1_name"])
    team2_name = st.sidebar.text_input(label='Nom de la deuxième équipe', value=selected_team_info["team2_name"])
    st.sidebar.markdown('---')

    tab1, tab2, tab3 = st.tabs(["Guide d'utilisation", "Couleurs des Équipes", "Hyperparamètres & Détection"])
    with tab1:
        st.header('Détection des Joueurs')
        st.subheader('Fonctionnalités principales :', divider='blue')
        st.markdown("""
                    1. Détection des joueurs, arbitre et ballon.
                    2. Prédiction des équipes des joueurs.
                    3. Estimation des positions sur une carte tactique.
                    4. Suivi du ballon.
                    """)
        st.subheader("Comment utiliser l'application ?", divider='blue')
        st.markdown("""
                    1. Importer une vidéo via le menu latéral.
                    2. Saisir les noms des équipes.
                    3. Aller dans l'onglet "Couleurs des Équipes" pour sélectionner une image représentative.
                    4. Cliquer sur les joueurs pour sélectionner les couleurs.
                    5. Ajuster les hyperparamètres si nécessaire.
                    6. Lancer la détection.
                    7. Si l'option "Enregistrer le résultat" est cochée, la vidéo sera sauvegardée dans le dossier outputs.
                    """)
        st.write("Version 0.0.1")

    colors_dic, color_list_lab = create_colors_info(team1_name, st.session_state.get(f"{team1_name} P color", selected_team_info["team1_p_color"]), st.session_state.get(f"{team1_name} GK color", selected_team_info["team1_gk_color"]),
                                                     team2_name, st.session_state.get(f"{team2_name} P color", selected_team_info["team2_p_color"]), st.session_state.get(f"{team2_name} GK color", selected_team_info["team2_gk_color"]))

    with tab2:
        t1col1, t1col2 = st.columns([1,1])
        with t1col1:
            cap_temp = cv2.VideoCapture(tempf.name)
            frame_count = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_nbr = st.slider(label="Sélectionnez une image (frame)", min_value=1, max_value=frame_count, step=1, help="Choisissez l'image sur laquelle vous allez identifier les couleurs des équipes")
            cap_temp.set(cv2.CAP_PROP_POS_FRAMES, frame_nbr)
            success, frame = cap_temp.read()
            with st.spinner('Détection des joueurs sur l’image sélectionnée...'):
                results = model_players(frame, conf=0.7)
                bboxes = results[0].boxes.xyxy.cpu().numpy()
                labels = results[0].boxes.cls.cpu().numpy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections_imgs_list = []
                detections_imgs_grid = []
                padding_img = np.ones((80,60,3),dtype=np.uint8)*255
                for i, j in enumerate(list(labels)):
                    if int(j) == 0:
                        bbox = bboxes[i,:]                         
                        obj_img = frame_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                        obj_img = cv2.resize(obj_img, (60,80))
                        detections_imgs_list.append(obj_img)
                detections_imgs_grid.append([detections_imgs_list[i] for i in range(len(detections_imgs_list)//2)])
                detections_imgs_grid.append([detections_imgs_list[i] for i in range(len(detections_imgs_list)//2, len(detections_imgs_list))])
                if len(detections_imgs_list)%2 != 0:
                    detections_imgs_grid[0].append(padding_img)
                concat_det_imgs_row1 = cv2.hconcat(detections_imgs_grid[0])
                concat_det_imgs_row2 = cv2.hconcat(detections_imgs_grid[1])
                concat_det_imgs = cv2.vconcat([concat_det_imgs_row1,concat_det_imgs_row2])
            st.write("Joueurs détectés")
            value = streamlit_image_coordinates(concat_det_imgs, key="numpy")
            st.markdown('---')
            radio_options =[f"{team1_name} Couleur joueurs", f"{team1_name} Couleur gardien",f"{team2_name} Couleur joueurs", f"{team2_name} Couleur gardien"]
            active_color = st.radio(label="Sélectionner la couleur à extraire sur l'image ci-dessus", options=radio_options, horizontal=True,
                                    help="Choisissez la couleur à extraire en cliquant sur un joueur. Les couleurs sont affichées ci-dessous.")
            if value is not None:
                picked_color = concat_det_imgs[value['y'], value['x'], :]
                st.session_state[f"{active_color}"] = '#%02x%02x%02x' % tuple(picked_color)
            st.write("Boîtes de sélection ci-dessous pour ajustement manuel des couleurs.")
            cp1, cp2, cp3, cp4 = st.columns([1,1,1,1])
            with cp1:
                hex_color_1 = st.session_state.get(f"{team1_name} Couleur joueurs", selected_team_info["team1_p_color"])
                team1_p_color = st.color_picker(label=' ', value=hex_color_1, key='t1p')
                st.session_state[f"{team1_name} Couleur joueurs"] = team1_p_color
            with cp2:
                hex_color_2 = st.session_state.get(f"{team1_name} Couleur gardien", selected_team_info["team1_gk_color"])
                team1_gk_color = st.color_picker(label=' ', value=hex_color_2, key='t1gk')
                st.session_state[f"{team1_name} Couleur gardien"] = team1_gk_color
            with cp3:
                hex_color_3 = st.session_state.get(f"{team2_name} Couleur joueurs", selected_team_info["team2_p_color"])
                team2_p_color = st.color_picker(label=' ', value=hex_color_3, key='t2p')
                st.session_state[f"{team2_name} Couleur joueurs"] = team2_p_color
            with cp4:
                hex_color_4 = st.session_state.get(f"{team2_name} Couleur gardien", selected_team_info["team2_gk_color"])
                team2_gk_color = st.color_picker(label=' ', value=hex_color_4, key='t2gk')
                st.session_state[f"{team2_name} Couleur gardien"] = team2_gk_color
        with t1col2:
            extracted_frame = st.empty()
            extracted_frame.image(frame, use_column_width=True, channels="BGR")

    with tab3:
        t2col1, t2col2 = st.columns([1,1])
        with t2col1:
            player_model_conf_thresh = st.slider('Seuil de confiance pour la détection des joueurs', min_value=0.0, max_value=1.0, value=0.6)
            keypoints_model_conf_thresh = st.slider('Seuil de confiance pour les points du terrain', min_value=0.0, max_value=1.0, value=0.7)
            keypoints_displacement_mean_tol = st.slider('Tolérance de déplacement des points (pixels)', min_value=-1, max_value=100, value=7,
                                                         help="Distance moyenne maximale tolérée entre les positions des points terrain détectés sur deux images consécutives.")
            detection_hyper_params = {
                0: player_model_conf_thresh,
                1: keypoints_model_conf_thresh,
                2: keypoints_displacement_mean_tol
            }
        with t2col2:
            num_pal_colors = st.slider(label="Nombre de couleurs à extraire", min_value=1, max_value=5, step=1, value=3,
                                    help="Nombre de couleurs extraites par joueur pour aider à la prédiction d'équipe.")
            st.markdown("---")
            save_output = st.checkbox(label='Enregistrer le résultat', value=False)
            if save_output:
                output_file_name = st.text_input(label='Nom du fichier (optionnel)', placeholder='Saisir un nom de fichier vidéo de sortie.')
            else:
                output_file_name = None
        st.markdown("---")

        bcol1, bcol2 = st.columns([1,1])
        with bcol1:
            nbr_frames_no_ball_thresh = st.number_input("Seuil de réinitialisation du suivi du ballon (frames)", min_value=1, max_value=10000,
                                                     value=30, help="Nombre de frames sans détection du ballon avant réinitialisation du suivi")
            ball_track_dist_thresh = st.number_input("Distance maximale entre détections consécutives du ballon (pixels)", min_value=1, max_value=1280,
                                                        value=100)
            max_track_length = st.number_input("Longueur maximale du suivi du ballon (nombre de détections)", min_value=1, max_value=1000,
                                                        value=35)
            ball_track_hyperparams = {
                0: nbr_frames_no_ball_thresh,
                1: ball_track_dist_thresh,
                2: max_track_length
            }
        with bcol2:
            st.write("Options d’annotation :")
            bcol21t, bcol22t = st.columns([1,1])
            with bcol21t:
                show_k = st.toggle(label="Afficher les points terrain", value=False)
                show_p = st.toggle(label="Afficher les joueurs", value=True)
            with bcol22t:
                show_pal = st.toggle(label="Afficher les palettes de couleurs", value=True)
                show_b = st.toggle(label="Afficher la trajectoire du ballon", value=True)
            plot_hyperparams = {
                0: show_k,
                1: show_pal,
                2: show_b,
                3: show_p
            }
            st.markdown('---')
            bcol21, bcol22, bcol23, bcol24 = st.columns([1.5,1,1,1])
            with bcol21:
                st.write('')
            with bcol22:
                ready = True if (team1_name == '') or (team2_name == '') else False
                start_detection = st.button(label='Lancer la Détection', disabled=ready)
            with bcol23:
                stop_btn_state = True if not start_detection else False
                stop_detection = st.button(label='Arrêter la Détection', disabled=stop_btn_state)
            with bcol24:
                st.write('')

        stframe = st.empty()
    cap = cv2.VideoCapture(tempf.name)
    status = False

    stframe = st.empty()
    cap = cv2.VideoCapture(tempf.name)
    status = False
    player_positions = []

    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

    if start_detection and not stop_detection:
        st.toast(f'Détection en cours...')
        status, player_positions = detect(cap, stframe, output_file_name, save_output, model_players, model_keypoints,
                                  detection_hyper_params, ball_track_hyperparams, plot_hyperparams,
                                  num_pal_colors, colors_dic, color_list_lab, tracker)

    else:
        try:
            cap.release()
        except:
            pass

    if status:
        st.toast(f'Détection terminée !')
        cap.release()

        import seaborn as sns
        import matplotlib.pyplot as plt

        if player_positions:
            st.subheader("Carte de chaleur des déplacements des joueurs", divider='orange')
            positions = np.array(player_positions)
            x = positions[:, 0]
            y = positions[:, 1]

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.kdeplot(x=x, y=y, cmap="Reds", fill=True, thresh=0.05, bw_adjust=0.5, ax=ax)
            ax.set_title("Heatmap des déplacements des joueurs", fontsize=16)
            ax.invert_yaxis()  # Pour correspondre à la vue vidéo
            ax.set_xlabel("Position horizontale (pixels)")
            ax.set_ylabel("Position verticale (pixels)")
            st.pyplot(fig)
        else:
            st.info("Aucune position de joueur n'a été enregistrée pour générer une heatmap.")



if __name__=='__main__':
    try:
        main()
    except SystemExit:
        pass

