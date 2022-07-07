from collections import defaultdict
from .recommender import Recommender
from .random import Random
import random
import heapq


class BlacklistRecommender(Recommender):
    """
    Recommend tracks closest to the previous one.
    Fall back to the random recommender if no
    recommendations found for the track.
    """

    def __init__(self, tracks_redis, recommendations_redis, catalog, artists_timeout, tracks_timeout, time_threshold):
        self.tracks_redis = tracks_redis
        self.recommendations_redis = recommendations_redis
        self.fallback = Random(tracks_redis)
        self.catalog = catalog

        self.recommend_buffer = defaultdict(list)
        self.artist_blacklist = defaultdict(list)
        self.tracks_blacklist = defaultdict(list)

        self.artists_timeout = artists_timeout
        self.tracks_timeout = tracks_timeout
        self.time_threshold = time_threshold

    def update_blacklist(self, user: int, track):
        self.artist_blacklist[user].append(track.artist)
        if len(self.artist_blacklist[user]) > self.artists_timeout:
            self.artist_blacklist[user] = self.artist_blacklist[user][1:]

        self.tracks_blacklist[user].append(track.track)
        if len(self.tracks_blacklist[user]) > self.tracks_timeout:
            self.tracks_blacklist[user] = self.tracks_blacklist[user][1:]

    def in_blacklist(self, user: int, track):
        return track.artist in self.artist_blacklist[user] or track.track in self.tracks_blacklist[user]

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        track_data = self.tracks_redis.get(prev_track)
        if track_data is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        if len(self.recommend_buffer[user]) == 0:
            heapq.heappush(self.recommend_buffer[user], (-prev_track_time, prev_track))

        track = self.catalog.from_bytes(track_data)
        if track is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        # if previous recommendation is bad
        if prev_track_time < self.time_threshold:
            self.update_blacklist(user, track)
            # clean user history until we meet another artist
            if user in self.recommend_buffer:
                while len(self.recommend_buffer[user]) and self.in_blacklist(user, track):
                    prev_track_time, prev_track = heapq.heappop(self.recommend_buffer[user])
                    track_data = self.tracks_redis.get(prev_track)
                    track = self.catalog.from_bytes(track_data)

        recommendations = track.recommendations
        if recommendations is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        shuffled = list(recommendations)
        random.shuffle(shuffled)

        # drop bad artists and tracks
        for next_track in shuffled:
            track_data = self.tracks_redis.get(next_track)
            track = self.catalog.from_bytes(track_data)
            if not self.in_blacklist(user, track):
                return next_track

        return self.fallback.recommend_next(user, prev_track, prev_track_time)
