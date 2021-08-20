# import time
# from queue import Queue
# from threading import Thread
#
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# from fluids.envs.environment_runner import EnvStepInfo, EnvironmentPlayer
# from fluids.envs.fluid_simulator import SimulatorConfig
# from helper_functions.misc_utils import format_float_array_values
# from models.dot_dict import DotDict
#
#
# class DebugPlotter(object):
#     DEFAULT_CFG = {'debug_text', 'debug_reward'}
#
#     def __init__(self, sim_config: SimulatorConfig, draw_config=DEFAULT_CFG):
#         self.cache = DotDict()
#         self.config = sim_config
#         self.draw_config = draw_config
#         self.cancel_current_draw = None
#
#     def __clear_if_present(self, key):
#         if key in self.cache:
#             self.cache[key].cla()
#
#     def _use_axis(self, name):
#         return self.draw_config and name in self.draw_config
#
#     def init_cache(self):
#         if 'debug_fig' not in self.cache:
#             fig_shape = (2, 3)
#             self.cache.debug_fig = plt.figure(figsize=(12, 6))
#
#             if self._use_axis("debug_text"):
#                 self.cache.debug_text = plt.subplot2grid(fig_shape, (0, 0), colspan=1)
#             if self._use_axis("debug_hist"):
#                 self.cache.debug_hist = plt.subplot2grid(fig_shape, (0, 2), colspan=1)
#             if self._use_axis("debug_q_vals"):
#                 self.cache.debug_q_vals = plt.subplot2grid(fig_shape, (0, 1), colspan=1, rowspan=1)
#             if self._use_axis("debug_reward"):
#                 self.cache.debug_reward = plt.subplot2grid(fig_shape, (0, 1), colspan=1)
#             if self._use_axis("debug_device"):
#                 self.cache.debug_device = plt.subplot2grid(fig_shape, (1, 0), colspan=1, rowspan=1)
#             if self._use_axis("debug_action"):
#                 self.cache.debug_action = plt.subplot2grid(fig_shape, (1, 1), colspan=2, rowspan=1)
#
#             self.cache.debug_fig.canvas.mpl_connect('close_event', self.cancel_current_draw)
#             self.cache.rewards = []
#             self.cache.rewards_mean = []
#
#     def clear_plot(self):
#         if "debug_fig" in self.cache:
#             self.__clear_if_present("debug_text")
#             self.__clear_if_present("debug_hist")
#             self.__clear_if_present("debug_q_vals")
#             self.__clear_if_present("debug_reward")
#             self.__clear_if_present("debug_device")
#             self.__clear_if_present("debug_action")
#
#     def cached(self, key):
#         return self.cache.get(key, None)
#
#     def on_agent_step(self, env_step_info: EnvStepInfo):
#         pass
#
#     def on_episode_begin(self, episode_id):
#         pass
#
#     def on_episode_end(self, episode_id):
#         pass
#
#
# class ThreadedPlotter(DebugPlotter):
#     def __init__(self, sim_config, show_episodes=None, do_async=True, draw_config=DebugPlotter.DEFAULT_CFG):
#         super().__init__(sim_config, draw_config=draw_config)
#         self.show_episodes = show_episodes
#         self.player = None
#         self.stop_cmd = False
#
#         self.worker_queue = Queue()
#         self.do_async = do_async
#         self.draw_finished = False
#         if self.do_async:
#             self.worker_thread = Thread(target=self.work)
#             self.worker_thread.daemon = False
#             self.worker_thread.start()
#         self.init_cache()
#
#     def set_player(self, player: EnvironmentPlayer):
#         self.player = player
#
#     def work(self):
#
#
#         while True:
#             if self.worker_queue.empty():
#                 if self.stop_cmd:
#                     break
#                 time.sleep(0.01)
#             else:
#                 while False and self.worker_queue.qsize() > 1:
#                     self.worker_queue.get_nowait()
#                 try:
#                     self.clear_plot()
#                     self.draw_finished = self.draw_async(self.worker_queue.get_nowait())
#                     if self.draw_finished:
#                         self.on_figure_ready(self.cache.debug_fig)
#
#                     if self.stop_cmd:
#                         print("\r", "processing queue", self.worker_queue.qsize(), "remaining")
#                 except Exception:
#                     print("exception on drawing")
#                     self.draw_finished = False
#
#     def join(self):
#         self.stop_cmd = True
#         if self.worker_thread:
#             self.worker_thread.join()
#
#     def on_figure_ready(self, fig):
#         plt.pause(0.00000001)
#
#     def on_agent_step(self, env_step_info: EnvStepInfo):
#         if self.cancel_current_draw:
#             print("cancel draw")
#             return
#
#         if self.draw_finished:
#             print("drawfinished:", self.draw_finished)
#             self.draw_finished = False
#
#         if not any([self.show_episodes is True,
#                     isinstance(self.show_episodes, list) and env_step_info.episode in self.show_episodes]):
#             return True
#
#         if self.do_async:
#             self.worker_queue.put(env_step_info)
#         else:
#             self.draw_finished = self.draw_async(env_step_info)
#             if self.draw_finished:
#                 self.on_figure_ready(self.cache.debug_fig)
#
#     def draw_action(self, action_ax, esi):
#         data_labels = []
#         data = []
#         chan_dict = {0: "r", 1: "b"}
#         for key in esi.env.action_handler.action_dict.keys():
#             action = esi.env.action_handler.action_dict[key]
#             chan, inf = action['channel'], action['inf']
#
#             data_labels.append("{}:{}".format(chan_dict[chan], inf))
#             data.append(0)
#
#         data[esi.action] = 1
#
#         bars = action_ax.bar(data_labels, data, color="b")
#         bars[esi.action].set_color('b')
#
#     def draw_data_table(self, table_ax, esi):
#
#         row_labels = ['target', 'actual', 'infuse', 'dist']
#         column_labels = ['R', 'W']
#
#         state = esi.state
#         # print("state", state, "action", esi.action)
#         # print("next state", esi.next_state)
#
#         g_r = state[0]
#         s_r = state[-1]
#         inf_r, inf_w = state[1:3]
#
#         dist = g_r - s_r
#
#         table_data = [format_float_array_values([g_r, 0]),
#                       format_float_array_values([s_r, 0]),
#                       format_float_array_values([inf_r, inf_w]),
#                       format_float_array_values([dist, 0])]
#
#         table_ax.table(rowLabels=row_labels, colLabels=column_labels,
#                        cellText=table_data,
#                        loc="center", colWidths=[0.2 for x in column_labels],
#                        fontsize=55, cellLoc="center", edges="vertical")
#
#         self.cache.debug_text.axis('off')
#
#     def draw_agent_vis(self, agent_ax, esi):
#
#         self.player.visualize_agent(agent_ax, esi)
#
#     def draw_hist(self, hist_ax, esi, device):
#         colors = ['r', 'g', 'b']
#         for idx, chan in enumerate(colors):
#             hist_ax.hist(device[:, :, idx].flatten().round().astype(np.uint8), 256,
#                          histtype='bar',
#                          orientation='vertical', color=chan, alpha=0.85, label=chan, density=True)
#
#     def draw_device(self, device_ax, esi):
#
#         actual_device = esi.env.render('real')[0].round().astype(np.uint8)
#         device_ax.imshow(actual_device)
#         device_ax.set_title("actual state")
#         return actual_device
#
#     def draw_rewards(self, reward_ax, esi):
#         self.cache.rewards.append(esi.reward)
#
#         reward_ax.plot(
#             [i * self.config.SIMULATION_SECONDS_STEPS for i in range(len(self.cache.rewards))],
#             self.cache.rewards, color="b", label="reward")
#         reward_ax.plot(self.cache.rewards_mean, color="g", label="reward_mean")
#         reward_ax.legend()
#
#     def draw_async(self, esi: EnvStepInfo):
#         try:
#             if self._use_axis("debug_text"):
#                 self.cache.debug_text.set_title(self.player.name() + "@" + str(esi.step))
#                 self.draw_data_table(self.cache.debug_text, esi)
#
#             if self._use_axis("debug_reward"):
#                 self.draw_rewards(self.cache.debug_reward, esi)
#             if self._use_axis("debug_device"):
#                 device = self.draw_device(self.cache.debug_device, esi)
#                 if self._use_axis("debug_hist"):
#                     self.draw_hist(self.cache.debug_hist, esi, device)
#
#             if self._use_axis("debug_q_vals"):
#                 self.draw_agent_vis(self.cache.debug_q_vals, esi)
#             if self._use_axis("debug_action"):
#                 self.draw_action(self.cache.debug_action, esi)
#
#             return True
#
#         except Exception as e:
#             print("error", e)
#
#         return False
#
#
# class Plotter1D(ThreadedPlotter):
#
#     def draw_hist(self, hist_ax, esi, device):
#         colors = ['r', 'g', 'b']
#         for idx, chan in enumerate(colors):
#             hist_ax.hist(device[:, :, idx].flatten().round().astype(np.uint8), 256,
#                          histtype='bar',
#                          orientation='vertical', color=chan, alpha=0.85, label=chan, density=True)
#
#     def draw_device(self, device_ax, esi):
#         actual_device = esi.env.render('real')[0].round().astype(np.uint8)
#         device_ax.imshow(actual_device)
#         device_ax.set_title("actual state")
#         return actual_device
#
#     def draw_rewards(self, reward_ax, esi):
#         self.cache.rewards.append(esi.reward)
#
#         reward_ax.plot(
#             [i * self.config.SIMULATION_SECONDS_STEPS for i in range(len(self.cache.rewards))],
#             self.cache.rewards, color="b", label="reward")
#         reward_ax.plot(self.cache.rewards_mean, color="g", label="reward_mean")
#         reward_ax.legend()
#
#     def draw_data_table(self, table_ax, esi):
#         row_labels = ['target', 'actual', 'infuse', 'dist']
#         column_labels = ['R', 'W']
#
#         state = esi.state
#         # print("state", state, "action", esi.action)
#         # print("next state", esi.next_state)
#
#         g_r = state[0]
#         s_r = state[-1]
#         inf_r, inf_w = state[1:3]
#
#         dist = g_r - s_r
#
#         table_data = [format_float_array_values([g_r, 0]),
#                       format_float_array_values([s_r, 0]),
#                       format_float_array_values([inf_r, inf_w]),
#                       format_float_array_values([dist, 0])]
#
#         table_ax.table(rowLabels=row_labels, colLabels=column_labels,
#                        cellText=table_data,
#                        loc="center", colWidths=[0.2 for x in column_labels],
#                        fontsize=55, cellLoc="center", edges="vertical")
#
#         self.cache.debug_text.axis('off')
#
#     def on_figure_ready(self, fig):
#         print("figure rdy")
#         self.on_new_figure(fig)
#
#     def on_new_figure(self, fig):
#         print("new figure")
#
#         fig.canvas.draw()
#
#         width, height = fig.get_size_inches() * fig.get_dpi()
#         width = int(width)
#         height = int(height)
#         mplimage = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape((height, width, 3))
#
#         # self.write_text(img, str(measure_str))
#         # self.write_text(img, "{0:.3f}".format(ep_str), v_offset=25)
#         # self.write_text(img, "g:{0:.3f}".format(s[0]), v_offset=45)
#         # self.write_text(img, str(format_float_array_values(s[1:3].tolist())), v_offset=65)
#
#         cv2.imshow("", mplimage)
#         cv2.waitKey(1)
#
#
# class PlotterRGB(ThreadedPlotter):
#
#     def draw_data_table(self, table_ax, esi):
#         row_labels = ['target', 'actual', 'infuse', 'dist']
#         column_labels = ['R', 'G', 'B', 'W']
#
#         state = esi.state
#         # print("state", state, "action", esi.action)
#         # print("next state", esi.next_state)
#
#         g_r = state[0:3]
#         s_r = state[-3:]
#         inf_r, inf_g, inf_b, inf_w = state[3:7]
#
#         dist = g_r - s_r
#
#         table_data = [format_float_array_values(g_r.tolist() + [0]),
#                       format_float_array_values(s_r.tolist() + [0]),
#                       format_float_array_values([inf_r, inf_g, inf_b, inf_w]),
#                       format_float_array_values(dist.tolist() + [0])]
#
#         table_ax.table(rowLabels=row_labels, colLabels=column_labels,
#                        cellText=table_data,
#                        loc="center", colWidths=[0.2 for x in column_labels],
#                        fontsize=55, cellLoc="center", edges="vertical")
#
#         self.cache.debug_text.set_title(self.player.name())
#         self.cache.debug_text.axis('off')
