# Import libraries
import numpy as np
import pandas as pd
import streamlit as st
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import skimage
from PIL import Image, ImageColor
from ultralytics import YOLO
from sklearn.metrics import mean_squared_error

import os
import json
import yaml
import time



def get_labels_dics():
    # Get tactical map keypoints positions dictionary
    json_path = "../pitch map labels position.json"
    with open(json_path, 'r') as f:
        keypoints_map_pos = json.load(f)

    # Get football field keypoints numerical to alphabetical mapping
    yaml_path = "../config pitch dataset.yaml"
    with open(yaml_path, 'r') as file:
        classes_names_dic = yaml.safe_load(file)
    classes_names_dic = classes_names_dic['names']

    # Get football field keypoints numerical to alphabetical mapping
    yaml_path = "../config players dataset.yaml"
    with open(yaml_path, 'r') as file:
        labels_dic = yaml.safe_load(file)
    labels_dic = labels_dic['names']
    return keypoints_map_pos, classes_names_dic, labels_dic

def create_colors_info(team1_name, team1_p_color, team1_gk_color, team2_name, team2_p_color, team2_gk_color):
    team1_p_color_rgb = ImageColor.getcolor(team1_p_color, "RGB")
    team1_gk_color_rgb = ImageColor.getcolor(team1_gk_color, "RGB")
    team2_p_color_rgb = ImageColor.getcolor(team2_p_color, "RGB")
    team2_gk_color_rgb = ImageColor.getcolor(team2_gk_color, "RGB")

    colors_dic = {
        team1_name:[team1_p_color_rgb, team1_gk_color_rgb],
        team2_name:[team2_p_color_rgb, team2_gk_color_rgb]
    }
    colors_list = colors_dic[team1_name]+colors_dic[team2_name] # Define color list to be used for detected player team prediction
    color_list_lab = [skimage.color.rgb2lab([i/255 for i in c]) for c in colors_list] # Converting color_list to L*a*b* space
    return colors_dic, color_list_lab

def generate_file_name():
    list_video_files = os.listdir('./outputs/')
    idx = 0
    while True:
        idx +=1
        output_file_name = f'detect_{idx}'
        if output_file_name+'.mp4' not in list_video_files:
            break
    return output_file_name

def detect(cap, stframe, output_file_name, save_output, model_players, model_keypoints,
            hyper_params, ball_track_hyperparams, plot_hyperparams, num_pal_colors, colors_dic, color_list_lab, tracker):
    
    # Dictionnaire pour stocker les votes de chaque ID de joueur
    player_id_to_votes = {}
    player_id_to_team = {}  # Pour mémoriser l'équipe finale une fois assez de votes sont collectés
    track_id_to_positions = {}
    player_positions = []
    track_id_to_team_idx = {}  # Mémorise une fois pour chaque track_id l'équipe prédite
    show_k = plot_hyperparams[0]
    show_pal = plot_hyperparams[1]
    show_b = plot_hyperparams[2]
    show_p = plot_hyperparams[3]

    p_conf = hyper_params[0]
    k_conf = hyper_params[1]
    k_d_tol = hyper_params[2]

    nbr_frames_no_ball_thresh = ball_track_hyperparams[0]
    ball_track_dist_thresh = ball_track_hyperparams[1]
    max_track_length = ball_track_hyperparams[2]
    nbr_team_colors = len(list(colors_dic.values())[0])

    if not output_file_name:
        output_file_name = generate_file_name()
    csv_path = f"player_heatmaps/player_summary_{output_file_name}.csv"



    # Read tactical map image
    tac_map = cv2.imread('../tactical map.jpg')
    tac_width = tac_map.shape[0]
    tac_height = tac_map.shape[1]
    
    # Create output video writer
    if save_output:
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + tac_width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + tac_height
        output = cv2.VideoWriter(f'./outputs/{output_file_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))

    # Create progress bar
    tot_nbr_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st_prog_bar = st.progress(0, text='Detection starting.')

    keypoints_map_pos, classes_names_dic, labels_dic = get_labels_dics()

    # Set variable to record the time when we processed last frame 
    prev_frame_time = 0
    # Set variable to record the time at which we processed current frame 
    new_frame_time = 0
    
    # Store the ball track history
    ball_track_history = {'src':[],
                          'dst':[]
    }

    nbr_frames_no_ball = 0

    

    # Loop over input video frames
    for frame_nbr in range(1, tot_nbr_frames+1):

        # Update progress bar
        percent_complete = int(frame_nbr/(tot_nbr_frames)*100)
        st_prog_bar.progress(percent_complete, text=f"Detection in progress ({percent_complete}%)")

        # Read a frame from the video
        success, frame = cap.read()

        # Reset tactical map image for each new frame
        tac_map_copy = tac_map.copy()

        if nbr_frames_no_ball>nbr_frames_no_ball_thresh:
            ball_track_history['dst'] = []
            ball_track_history['src'] = []

        if success:

            #################### Part 1 ####################
            # Object Detection & Coordiante Transofrmation #
            ################################################

            # Run YOLOv8 players inference on the frame
            results_players = model_players(frame, conf=p_conf)
            # Run YOLOv8 field keypoints inference on the frame
            results_keypoints = model_keypoints(frame, conf=k_conf)
            
            

            ## Extract detections information
            bboxes_p = results_players[0].boxes.xyxy.cpu().numpy()                          # Detected players, referees and ball (x,y,x,y) bounding boxes
            bboxes_p_c = results_players[0].boxes.xywh.cpu().numpy()                        # Detected players, referees and ball (x,y,w,h) bounding boxes    
            labels_p = list(results_players[0].boxes.cls.cpu().numpy())                     # Detected players, referees and ball labels list
            confs_p = list(results_players[0].boxes.conf.cpu().numpy())                     # Detected players, referees and ball confidence level
            
            bboxes_k = results_keypoints[0].boxes.xyxy.cpu().numpy()                        # Detected field keypoints (x,y,x,y) bounding boxes
            bboxes_k_c = results_keypoints[0].boxes.xywh.cpu().numpy()                      # Detected field keypoints (x,y,w,h) bounding boxes
            labels_k = list(results_keypoints[0].boxes.cls.cpu().numpy())                   # Detected field keypoints labels list
            # Filtrer uniquement les joueurs (label == 0)
            player_indices = [i for i, lbl in enumerate(labels_p) if lbl == 0]
            player_bboxes = bboxes_p[player_indices]
            player_confs = [confs_p[i] for i in player_indices]

            # Format attendu : [[x, y, w, h], ...]
            tracker_inputs = []
            for i, box in enumerate(player_bboxes):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                tracker_inputs.append(([x1, y1, w, h], player_confs[i], 'player'))

            # Met à jour les objets suivis
            tracks = tracker.update_tracks(tracker_inputs, frame=frame)

            

            # Convert detected numerical labels to alphabetical labels
            detected_labels = [classes_names_dic[i] for i in labels_k]

            # Extract detected field keypoints coordiantes on the current frame
            detected_labels_src_pts = np.array([list(np.round(bboxes_k_c[i][:2]).astype(int)) for i in range(bboxes_k_c.shape[0])])

            # Get the detected field keypoints coordinates on the tactical map
            detected_labels_dst_pts = np.array([keypoints_map_pos[i] for i in detected_labels])


            ## Calculate Homography transformation matrix when more than 4 keypoints are detected
            if len(detected_labels) > 3:
                # Always calculate homography matrix on the first frame
                if frame_nbr > 1:
                    # Determine common detected field keypoints between previous and current frames
                    common_labels = set(detected_labels_prev) & set(detected_labels)
                    # When at least 4 common keypoints are detected, determine if they are displaced on average beyond a certain tolerance level
                    if len(common_labels) > 3:
                        common_label_idx_prev = [detected_labels_prev.index(i) for i in common_labels]   # Get labels indexes of common detected keypoints from previous frame
                        common_label_idx_curr = [detected_labels.index(i) for i in common_labels]        # Get labels indexes of common detected keypoints from current frame
                        coor_common_label_prev = detected_labels_src_pts_prev[common_label_idx_prev]     # Get labels coordiantes of common detected keypoints from previous frame
                        coor_common_label_curr = detected_labels_src_pts[common_label_idx_curr]          # Get labels coordiantes of common detected keypoints from current frame
                        coor_error = mean_squared_error(coor_common_label_prev, coor_common_label_curr)  # Calculate error between previous and current common keypoints coordinates
                        update_homography = coor_error > k_d_tol                                         # Check if error surpassed the predefined tolerance level
                    else:
                        update_homography = True                                                         
                else:
                    update_homography = True

                if  update_homography:
                    homog, mask = cv2.findHomography(detected_labels_src_pts,                   # Calculate homography matrix
                                                detected_labels_dst_pts)                  
            if 'homog' in locals():
                detected_labels_prev = detected_labels.copy()                               # Save current detected keypoint labels for next frame
                detected_labels_src_pts_prev = detected_labels_src_pts.copy()               # Save current detected keypoint coordiantes for next frame

                bboxes_p_c_0 = bboxes_p_c[[i==0 for i in labels_p],:]                       # Get bounding boxes information (x,y,w,h) of detected players (label 0)
                bboxes_p_c_2 = bboxes_p_c[[i==2 for i in labels_p],:]                       # Get bounding boxes information (x,y,w,h) of detected ball(s) (label 2)

                # Get coordinates of detected players on frame (x_cencter, y_center+h/2)
                detected_ppos_src_pts = bboxes_p_c_0[:,:2]  + np.array([[0]*bboxes_p_c_0.shape[0], bboxes_p_c_0[:,3]/2]).transpose()
                for pt in detected_ppos_src_pts:
                    player_positions.append([int(pt[0]), int(pt[1])])

                # Get coordinates of the first detected ball (x_center, y_center)
                detected_ball_src_pos = bboxes_p_c_2[0,:2] if bboxes_p_c_2.shape[0]>0 else None

                if detected_ball_src_pos is None:
                    nbr_frames_no_ball+=1
                else: 
                    nbr_frames_no_ball=0

                # Transform players coordinates from frame plane to tactical map plance using the calculated Homography matrix
                pred_dst_pts = []                                                           # Initialize players tactical map coordiantes list
                for pt in detected_ppos_src_pts:                                            # Loop over players frame coordiantes
                    pt = np.append(np.array(pt), np.array([1]), axis=0)                     # Covert to homogeneous coordiantes
                    dest_point = np.matmul(homog, np.transpose(pt))                              # Apply homography transofrmation
                    dest_point = dest_point/dest_point[2]                                   # Revert to 2D-coordiantes
                    pred_dst_pts.append(list(np.transpose(dest_point)[:2]))                 # Update players tactical map coordiantes list
                pred_dst_pts = np.array(pred_dst_pts)

                # Transform ball coordinates from frame plane to tactical map plance using the calculated Homography matrix
                if detected_ball_src_pos is not None:
                    pt = np.append(np.array(detected_ball_src_pos), np.array([1]), axis=0)
                    dest_point = np.matmul(homog, np.transpose(pt))
                    dest_point = dest_point/dest_point[2]
                    detected_ball_dst_pos = np.transpose(dest_point)

                    # track ball history
                    if show_b:
                        if len(ball_track_history['src'])>0 :
                            if np.linalg.norm(detected_ball_src_pos-ball_track_history['src'][-1])<ball_track_dist_thresh:
                                ball_track_history['src'].append((int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1])))
                                ball_track_history['dst'].append((int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1])))
                            else:
                                ball_track_history['src']=[(int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1]))]
                                ball_track_history['dst']=[(int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1]))]
                        else:
                            ball_track_history['src'].append((int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1])))
                            ball_track_history['dst'].append((int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1])))
                    
                if len(ball_track_history) > max_track_length:
                    ball_track_history['src'].pop(0)
                    ball_track_history['dst'].pop(0)

                
            ######### Part 2 ########## 
            # Players Team Prediction #
            ###########################

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                      # Convert frame to RGB
            obj_palette_list = []                                                                   # Initialize players color palette list
            palette_interval = (0,num_pal_colors)                                                   # Color interval to extract from dominant colors palette (1rd to 5th color)

            ## Loop over detected players (label 0) and extract dominant colors palette based on defined interval
            for i, j in enumerate(list(results_players[0].boxes.cls.cpu().numpy())):
                if int(j) == 0:
                    bbox = results_players[0].boxes.xyxy.cpu().numpy()[i,:]                         # Get bbox info (x,y,x,y)
                    obj_img = frame_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]       # Crop bbox out of the frame
                    obj_img_w, obj_img_h = obj_img.shape[1], obj_img.shape[0]
                    center_filter_x1 = np.max([(obj_img_w//2)-(obj_img_w//5), 1])
                    center_filter_x2 = (obj_img_w//2)+(obj_img_w//5)
                    center_filter_y1 = np.max([(obj_img_h//3)-(obj_img_h//5), 1])
                    center_filter_y2 = (obj_img_h//3)+(obj_img_h//5)
                    center_filter = obj_img[center_filter_y1:center_filter_y2, 
                                            center_filter_x1:center_filter_x2]
                    obj_pil_img = Image.fromarray(np.uint8(center_filter))                          # Convert to pillow image
                    reduced = obj_pil_img.convert("P", palette=Image.Palette.WEB)                   # Convert to web palette (216 colors)
                    palette = reduced.getpalette()                                                  # Get palette as [r,g,b,r,g,b,...]
                    palette = [palette[3*n:3*n+3] for n in range(256)]                              # Group 3 by 3 = [[r,g,b],[r,g,b],...]
                    color_count = [(n, palette[m]) for n,m in reduced.getcolors()]                  # Create list of palette colors with their frequency
                    RGB_df = pd.DataFrame(color_count, columns = ['cnt', 'RGB']).sort_values(       # Create dataframe based on defined palette interval
                                        by = 'cnt', ascending = False).iloc[
                                            palette_interval[0]:palette_interval[1],:]
                    palette = list(RGB_df.RGB)                                                      # Convert palette to list (for faster processing)
                    
                    # Update detected players color palette list
                    obj_palette_list.append(palette)
            
            ## Calculate distances between each color from every detected player color palette and the predefined teams colors
            players_distance_features = []
            # Loop over detected players extracted color palettes
            for palette in obj_palette_list:
                palette_distance = []
                palette_lab = [skimage.color.rgb2lab([i/255 for i in color]) for color in palette]  # Convert colors to L*a*b* space
                # Loop over colors in palette
                for color in palette_lab:
                    distance_list = []
                    # Loop over predefined list of teams colors
                    for c in color_list_lab:
                        #distance = np.linalg.norm([i/255 - j/255 for i,j in zip(color,c)])
                        distance = skimage.color.deltaE_cie76(color, c)                             # Calculate Euclidean distance in Lab color space
                        distance_list.append(distance)                                              # Update distance list for current color
                    palette_distance.append(distance_list)                                          # Update distance list for current palette
                players_distance_features.append(palette_distance)                                  # Update distance features list

            ## Predict detected players teams based on distance features
            players_teams_list = []
            # Loop over players distance features
            for distance_feats in players_distance_features:
                vote_list=[]
                # Loop over distances for each color 
                for dist_list in distance_feats:
                    team_idx = dist_list.index(min(dist_list))//nbr_team_colors                     # Assign team index for current color based on min distance
                    vote_list.append(team_idx)                                                      # Update vote voting list with current color team prediction
                players_teams_list.append(max(vote_list, key=vote_list.count))                      # Predict current player team by vote counting


            #################### Part 3 #####################
            # Updated Frame & Tactical Map With Annotations #
            #################################################
            
            ball_color_bgr = (0,0,255)                                                                          # Color (GBR) for ball annotation on tactical map
            j=0                                                                                                 # Initializing counter of detected players
            palette_box_size = 10                                                                               # Set color box size in pixels (for display)
            annotated_frame = frame                                                                             # Create annotated frame

            # Loop over all tracked players with DeepSORT
            # Définir votes_needed selon la longueur de la vidéo
            if tot_nbr_frames < 300:
                votes_needed = 3
            elif tot_nbr_frames < 800:
                votes_needed = 7
            else:
                votes_needed = 10
            tot_nbr_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for track in tracks:

                if not track.is_confirmed() or track.det_class != 'player':
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                # Centre bas du joueur
                x_center = int((x1 + x2) / 2)
                y_center = y2
                player_positions.append([x_center, y_center])

                if track_id not in track_id_to_positions:
                    track_id_to_positions[track_id] = []
                track_id_to_positions[track_id].append((x_center, y_center))

                # Matching plus tolérant sur le centre
                matched_palette_idx = None
                for idx, box in enumerate(results_players[0].boxes.xyxy.cpu().numpy()):
                    bx1, by1, bx2, by2 = box
                    center_det = ((bx1 + bx2) / 2, (by1 + by2) / 2)
                    if abs(center_det[0] - x_center) < 30 and abs(center_det[1] - y_center) < 30:
                        matched_palette_idx = idx
                        break

                # Étape 1 : accumulate votes
                if matched_palette_idx is not None and matched_palette_idx < len(players_teams_list):
                    vote_team_idx = players_teams_list[matched_palette_idx]
                    if track_id not in player_id_to_votes:
                        player_id_to_votes[track_id] = []
                    player_id_to_votes[track_id].append(vote_team_idx)

                    # Étape 2 : lock team if enough votes
                    if len(player_id_to_votes[track_id]) >= votes_needed and track_id not in player_id_to_team:
                        most_common = max(set(player_id_to_votes[track_id]), key=player_id_to_votes[track_id].count)
                        player_id_to_team[track_id] = most_common

                # Étape 3 : get final team
                if track_id in player_id_to_team:
                    team_idx = player_id_to_team[track_id]
                    team_name = list(colors_dic.keys())[team_idx]
                    color_rgb = colors_dic[team_name][0]
                    color_bgr = color_rgb[::-1]
                else:
                    color_bgr = (255, 255, 255)
                    team_name = "Inconnu"

                # Annoter la vidéo
                if show_p:
                    vote_count = len(player_id_to_votes.get(track_id, []))
                    text = f"ID {track_id} - {team_name} ({vote_count})"
                    annotated_frame = cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_bgr, 2)
                    annotated_frame = cv2.putText(annotated_frame, text, (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

                # Annoter la carte tactique
                if 'homog' in locals():
                    pt = np.array([x_center, y_center, 1])
                    mapped_pt = np.dot(homog, pt)
                    mapped_pt = mapped_pt / mapped_pt[2]
                    tac_map_copy = cv2.circle(tac_map_copy, (int(mapped_pt[0]), int(mapped_pt[1])), radius=5, color=color_bgr, thickness=-1)
                    tac_map_copy = cv2.putText(tac_map_copy, str(track_id), (int(mapped_pt[0]), int(mapped_pt[1]) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)

            if show_k:
                for i in range(bboxes_k.shape[0]):
                    annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_k[i,0]), int(bboxes_k[i,1])),  # Add bbox annotations with team colors
                                                (int(bboxes_k[i,2]), int(bboxes_k[i,3])), (0,0,0), 1)
            # Plot the tracks
            if len(ball_track_history['src'])>0:
                points = np.hstack(ball_track_history['dst']).astype(np.int32).reshape((-1, 1, 2))
                tac_map_copy = cv2.polylines(tac_map_copy, [points], isClosed=False, color=(0, 0, 100), thickness=2)

            
            
            # Combine annotated frame and tactical map in one image with colored border separation
            border_color = [255,255,255]                                                                        # Set border color (BGR)
            annotated_frame=cv2.copyMakeBorder(annotated_frame, 40, 10, 10, 10,                                 # Add borders to annotated frame
                                                cv2.BORDER_CONSTANT, value=border_color)
            tac_map_copy = cv2.copyMakeBorder(tac_map_copy, 70, 50, 10, 10, cv2.BORDER_CONSTANT,                # Add borders to tactical map 
                                            value=border_color)      
            tac_map_copy = cv2.resize(tac_map_copy, (tac_map_copy.shape[1], annotated_frame.shape[0]))          # Resize tactical map
            final_img = cv2.hconcat((annotated_frame, tac_map_copy))                                            # Concatenate both images
            ## Add info annotation
            cv2.putText(final_img, "Tactical Map", (1370,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

            new_frame_time = time.time()                                                                        # Get time after finished processing current frame
            fps = 1/(new_frame_time-prev_frame_time)                                                            # Calculate FPS as 1/(frame proceesing duration)
            prev_frame_time = new_frame_time                                                                    # Save current time to be used in next frame
            cv2.putText(final_img, "FPS: " + str(int(fps)), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            
            # Display the annotated frame
            stframe.image(final_img, channels="BGR")
            #cv2.imshow("YOLOv8 Inference", frame)
            if save_output:
                output.write(cv2.resize(final_img, (width, height)))

    analyze_performance(track_id_to_positions, output_file_name)
    plot_speed_curve(track_id_to_positions, output_file_name)

    st_prog_bar.empty()  
    import glob
    for img_path in sorted(glob.glob(f"player_heatmaps/heatmap_*_{output_file_name}.png")):
        st.image(img_path, caption=os.path.basename(img_path))
    for img_path in sorted(glob.glob("player_heatmaps/speed_curve_*.png")):
        st.image(img_path, caption=os.path.basename(img_path))
    summary_df = build_summary_table(track_id_to_positions)
    st.dataframe(summary_df)

    # Afficher les 3 joueurs ayant parcouru le plus de distance
    top3 = summary_df.sort_values("Distance (px)", ascending=False).head(3)
    st.write("🏃‍♂️ Joueurs les plus actifs (distance parcourue) :")
    st.dataframe(top3[["Joueur", "Distance (px)"]])

    # Option de sélection de joueur
    if not summary_df.empty:
        selected_id = st.selectbox("👤 Sélectionner un joueur", summary_df["Joueur"], key="select_joueur_1")
        st.write("📊 Statistiques du joueur sélectionné :")
        st.dataframe(summary_df[summary_df["Joueur"] == selected_id])

        # Afficher la heatmap et la courbe de vitesse du joueur sélectionné
        heatmap_path = f"player_heatmaps/heatmap_{selected_id}_{output_file_name}.png"
        speedcurve_path = f"player_heatmaps/speed_curve_{selected_id}_{output_file_name}.png"

        if os.path.exists(heatmap_path):
            st.image(heatmap_path, caption=f"Heatmap - Joueur {selected_id}")
        if os.path.exists(speedcurve_path):
            st.image(speedcurve_path, caption=f"Évolution de la vitesse - Joueur {selected_id}")
    # Sauvegarde du résumé dans un CSV
    summary_df.to_csv(csv_path, index=False)

    with st.expander("📊 Analyse des performances"):
        st.dataframe(summary_df)

        top3 = summary_df.sort_values("Distance (px)", ascending=False).head(3)
        st.write("🏃‍♂️ Joueurs les plus actifs (distance parcourue) :")
        st.dataframe(top3[["Joueur", "Distance (px)"]])

        if not summary_df.empty:
            selected_id = st.selectbox("👤 Sélectionner un joueur", summary_df["Joueur"], key="select_joueur_2")
            st.write("📊 Statistiques du joueur sélectionné :")
            st.dataframe(summary_df[summary_df["Joueur"] == selected_id])

            heatmap_path = f"player_heatmaps/heatmap_{selected_id}_{output_file_name}.png"
            speedcurve_path = f"player_heatmaps/speed_curve_{selected_id}_{output_file_name}.png"

            
            if os.path.exists(heatmap_path):
                st.image(heatmap_path, caption=f"Heatmap - Joueur {selected_id}")
            if os.path.exists(speedcurve_path):
                st.image(speedcurve_path, caption=f"Évolution de la vitesse - Joueur {selected_id}")

        # Téléchargement CSV
        summary_df.to_csv(csv_path, index=False)
        with open(csv_path, "rb") as f:
            st.download_button("📥 Télécharger les stats", f, file_name="player_stats.csv", mime="text/csv")






    return True, player_positions


import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_performance(track_id_to_positions, output_file_name, field_shape=(1080, 1920)):
    os.makedirs("player_heatmaps", exist_ok=True)
    for track_id, positions in track_id_to_positions.items():
        positions = np.array(positions)
        xs, ys = positions[:, 0], positions[:, 1]

        # Calcul de la distance parcourue
        dists = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        total_distance = np.sum(dists)
        
        # Temps dans chaque moitié
        midline = field_shape[1] // 2
        time_left = np.sum(xs < midline)
        time_right = np.sum(xs >= midline)

        # Vitesse moyenne (en pixels/frame)
        avg_speed = total_distance / len(positions) if len(positions) > 1 else 0

        print(f"Joueur {track_id} :")
        print(f"- Distance parcourue : {total_distance:.2f} px")
        print(f"- Vitesse moyenne : {avg_speed:.2f} px/frame")
        print(f"- Temps à gauche : {time_left} frames")
        print(f"- Temps à droite : {time_right} frames\n")

        # Heatmap
        heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=(50, 50), range=[[0, field_shape[1]], [0, field_shape[0]]])
        heatmap = heatmap.T

        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap, cmap="coolwarm")  
        plt.title(f"Zones d’influence - Joueur {track_id}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.gca().invert_yaxis()
        plt.savefig(f"player_heatmaps/heatmap_{track_id}_{output_file_name}.png")
        plt.close()
def plot_speed_curve(track_id_to_positions, output_file_name):
    for track_id, positions in track_id_to_positions.items():
        positions = np.array(positions)
        if len(positions) < 2:
            continue
        dists = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        speeds = dists  # en px/frame
        frames = np.arange(1, len(speeds) + 1)

        plt.figure(figsize=(10, 4))
        plt.plot(frames, speeds, label=f"ID {track_id}")
        plt.xlabel("Frame")
        plt.ylabel("Vitesse (px/frame)")
        plt.title(f"Évolution de la vitesse - Joueur {track_id}")
        plt.grid()
        plt.savefig(f"player_heatmaps/speed_curve_{track_id}_{output_file_name}.png")
        plt.close()

def build_summary_table(track_id_to_positions, field_shape=(1080, 1920)):
    data = []
    for track_id, positions in track_id_to_positions.items():
        positions = np.array(positions)
        if len(positions) < 2:
            continue
        xs, ys = positions[:, 0], positions[:, 1]
        dists = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        total_distance = np.sum(dists)
        avg_speed = total_distance / len(positions)
        midline = field_shape[1] // 2
        time_left = np.sum(xs < midline)
        time_right = np.sum(xs >= midline)
        data.append({
            "Joueur": track_id,
            "Distance (px)": round(total_distance, 2),
            "Vitesse moy (px/frame)": round(avg_speed, 2),
            "Temps gauche": time_left,
            "Temps droite": time_right,
            "Frames": len(positions)
        })
    return pd.DataFrame(data)

    


    