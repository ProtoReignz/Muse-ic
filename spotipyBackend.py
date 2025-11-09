import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import os

#import model output mood stuff --> change input mapping to refer to said moods for each playlist


class SpotifyPlaylistController:
    def __init__(self):
        """Initialize Spotify connection"""
        # Your credentials
        #fill this shit
        
        # Scopes needed for playback control
        scope = 'user-modify-playback-state user-read-playback-state user-read-currently-playing'
        
        # Authenticate
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope=scope
        ))
        
        # Load user's input mapping
        self.load_input_mapping()
    
    def load_input_mapping(self):
        """Load the user's custom input-to-playlist mapping"""
        if os.path.exists('input_mapping.json'):
            with open('input_mapping.json', 'r') as f:
                self.input_mapping = json.load(f)
        else:
            # Default mapping (user can customize this)
            self.input_mapping = {
                "button1": None,
                "button2": None,
                "button3": None,
                "sensor_low": None,
                "sensor_medium": None,
                "sensor_high": None
            }
            self.save_input_mapping()
    
    def save_input_mapping(self):
        """Save the input mapping to file"""
        with open('input_mapping.json', 'w') as f:
            json.dump(self.input_mapping, f, indent=2)
    
    def set_playlist_for_input(self, input_name, playlist_uri):
        """
        Map a hardware input to a specific playlist
        
        Args:
            input_name: Name of the input (e.g., "button1", "sensor_high")
            playlist_uri: Spotify playlist URI (e.g., "spotify:playlist:37i9dQZF1DXcBWIGoYBM5M")
                         or just the ID (e.g., "37i9dQZF1DXcBWIGoYBM5M")
        """
        # Normalize playlist URI
        if not playlist_uri.startswith('spotify:playlist:'):
            playlist_uri = f'spotify:playlist:{playlist_uri}'
        
        self.input_mapping[input_name] = playlist_uri
        self.save_input_mapping()
        print(f"✓ Mapped '{input_name}' to playlist {playlist_uri}")
    
    def handle_hardware_input(self, input_name):
        """
        Handle a hardware input and play the corresponding playlist
        
        Args:
            input_name: The name of the input that was triggered
        """
        if input_name not in self.input_mapping:
            print(f"⚠️ Unknown input: {input_name}")
            return
        
        playlist_uri = self.input_mapping[input_name]
        
        if playlist_uri is None:
            print(f"⚠️ No playlist mapped to '{input_name}'")
            return
        
        try:
            # Get available devices
            devices = self.sp.devices()
            
            if not devices['devices']:
                print("❌ No active Spotify devices found!")
                print("   Please open Spotify on your computer, phone, or speaker.")
                return
            
            # Use the first available device
            device_id = devices['devices'][0]['id']
            device_name = devices['devices'][0]['name']
            
            # Play the playlist
            self.sp.start_playback(
                device_id=device_id,
                context_uri=playlist_uri
            )
            
            print(f"🎵 Playing playlist on '{device_name}'")
            
            # Show what's playing
            current = self.sp.current_playback()
            if current and current['item']:
                track = current['item']['name']
                artist = current['item']['artists'][0]['name']
                print(f"   Now playing: {track} by {artist}")
        
        except Exception as e:
            print(f"❌ Error playing playlist: {e}")
    
    def search_playlists(self, query, limit=10):
        """
        Search for playlists by name
        
        Args:
            query: Search term
            limit: Number of results to return
        
        Returns:
            List of playlists with id, name, and uri
        """
        results = self.sp.search(q=query, type='playlist', limit=limit)
        playlists = []
        
        for item in results['playlists']['items']:
            playlists.append({
                'id': item['id'],
                'name': item['name'],
                'uri': item['uri'],
                'owner': item['owner']['display_name'],
                'tracks': item['tracks']['total']
            })
        
        return playlists
    
    def get_user_playlists(self):
        """Get the current user's saved playlists"""
        playlists = []
        results = self.sp.current_user_playlists()
        
        while results:
            for item in results['items']:
                playlists.append({
                    'id': item['id'],
                    'name': item['name'],
                    'uri': item['uri'],
                    'tracks': item['tracks']['total']
                })
            
            # Handle pagination
            if results['next']:
                results = self.sp.next(results)
            else:
                results = None
        
        return playlists
    
    def show_current_mapping(self):
        """Display the current input-to-playlist mapping"""
        print("\n=== Current Input Mapping ===")
        for input_name, playlist_uri in self.input_mapping.items():
            if playlist_uri:
                # Get playlist info
                try:
                    playlist_id = playlist_uri.split(':')[-1]
                    playlist = self.sp.playlist(playlist_id, fields='name')
                    print(f"  {input_name:20} → {playlist['name']}")
                except:
                    print(f"  {input_name:20} → {playlist_uri}")
            else:
                print(f"  {input_name:20} → (not set)")
        print("=============================\n")
    
    def get_available_devices(self):
        """List all available Spotify devices"""
        devices = self.sp.devices()
        return devices['devices']


# Example usage and setup function
def setup_playlist_mapping():
    """Interactive setup to map inputs to playlists"""
    controller = SpotifyPlaylistController()
    
    print("\n🎵 Spotify Playlist Controller Setup")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. View current mapping")
        print("2. Search for playlists")
        print("3. View your playlists")
        print("4. Map input to playlist")
        print("5. Test an input")
        print("6. View available devices")
        print("7. Exit")
        
        choice = input("\nChoose an option: ").strip()
        
        if choice == '1':
            controller.show_current_mapping()
        
        elif choice == '2':
            query = input("Search for playlist: ").strip()
            playlists = controller.search_playlists(query)
            
            print(f"\n🔍 Found {len(playlists)} playlists:")
            for i, pl in enumerate(playlists, 1):
                print(f"  {i}. {pl['name']} by {pl['owner']} ({pl['tracks']} tracks)")
                print(f"     URI: {pl['uri']}")
        
        elif choice == '3':
            playlists = controller.get_user_playlists()
            
            print(f"\n📚 Your playlists ({len(playlists)}):")
            for i, pl in enumerate(playlists, 1):
                print(f"  {i}. {pl['name']} ({pl['tracks']} tracks)")
                print(f"     URI: {pl['uri']}")
        
        elif choice == '4':
            print("\nAvailable inputs:")
            for i, input_name in enumerate(controller.input_mapping.keys(), 1):
                print(f"  {i}. {input_name}")
            
            input_name = input("\nEnter input name: ").strip()
            playlist_uri = input("Enter playlist URI or ID: ").strip()
            
            controller.set_playlist_for_input(input_name, playlist_uri)
        
        elif choice == '5':
            input_name = input("Enter input name to test: ").strip()
            controller.handle_hardware_input(input_name)
        
        elif choice == '6':
            devices = controller.get_available_devices()
            print(f"\n🔊 Available devices ({len(devices)}):")
            for device in devices:
                status = "🟢 Active" if device['is_active'] else "⚪ Inactive"
                print(f"  {status} {device['name']} ({device['type']})")
        
        elif choice == '7':
            print("Goodbye!")
            break


if __name__ == "__main__":
    setup_playlist_mapping()