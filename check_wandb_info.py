import requests
import json

# --- Configuration ---
API_KEY = '4eef6773b5e61175467b70de7dd1a5ac523bfdc8'
BASE_URL = 'https://api.wandb.ai/v1'

HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json',
}

def get_user_info():
    """Fetches user information using GraphQL, which includes the default entity."""
    graphql_url = 'https://api.wandb.ai/graphql' # Standard W&B GraphQL endpoint
    query = """
    query GetViewer {
      viewer {
        id
        username
        email
        entity
        teams {
          edges {
            node {
              name
            }
          }
        }
      }
    }
    """
    print(f"Fetching user info via GraphQL from: {graphql_url}")
    try:
        response = requests.post(graphql_url, json={'query': query}, headers=HEADERS)
        response.raise_for_status()  # Raises an exception for HTTP errors (4xx or 5xx)
        
        response_data = response.json()
        
        if 'errors' in response_data:
            print("GraphQL API returned errors:")
            for error in response_data['errors']:
                print(f"  - {error.get('message')}")
                if 'extensions' in error and 'code' in error['extensions']:
                    if error['extensions']['code'] == 'PERMISSION_DENIED' or \
                       error['extensions']['code'] == 'UNAUTHENTICATED':
                        print("  This suggests an issue with the API key (e.g., invalid or insufficient permissions).")
            return None

        user_data = response_data.get('data', {}).get('viewer')
        print("\n--- User Information (GraphQL) ---")
        if user_data:
            entity = user_data.get('entity')
            email = user_data.get('email')
            username = user_data.get('username')
            print(f"  Username: {username}")
            print(f"  Default Entity: {entity}")
            print(f"  Email: {email}")
            teams = user_data.get('teams', {}).get('edges', [])
            if teams:
                print("  Teams:")
                for team_edge in teams:
                    print(f"    - {team_edge.get('node', {}).get('name')}")
            return entity
        else:
            print("  No user data found in GraphQL response.")
            print(f"  Full response data: {response_data}")
            return None
            
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while fetching user info via GraphQL: {http_err}")
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred while fetching user info via GraphQL: {req_err}")
    except json.JSONDecodeError:
        print(f"Failed to decode JSON response from GraphQL endpoint. Response: {response.text}")
    return None

def get_projects(entity):
    """Fetches projects for a given entity."""
    if not entity:
        print("\nCannot fetch projects without an entity.")
        return

    url = f'{BASE_URL}/projects?entity={entity}'
    # Alternative endpoint that might list projects across entities the user has access to:
    # url = f'{BASE_URL}/projects' 
    print(f"\nFetching projects for entity '{entity}' from: {url}")
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        projects_data = response.json()
        print("\n--- Projects ---   ")
        if projects_data and 'edges' in projects_data:
            projects = projects_data['edges']
            if projects:
                for i, project_edge in enumerate(projects):
                    project_node = project_edge.get('node', {})
                    print(f"  {i+1}. Project Name: {project_node.get('name')}")
                    print(f"     Entity: {project_node.get('entityName')}")
                    print(f"     Description: {project_node.get('description', 'N/A')}")
                    print(f"     Runs Count: {project_node.get('runs', {}).get('totalCount', 'N/A')}")
                    print("     ----")
            else:
                print(f"  No projects found for entity '{entity}'.")
        elif projects_data and isinstance(projects_data, list): # Fallback for different API structure
             if projects_data:
                for i, project in enumerate(projects_data):
                    print(f"  {i+1}. Project Name: {project.get('name')}")
                    print(f"     Entity: {project.get('entity_name', entity)}") # Assuming entity_name or use passed entity
                    print("     ----")
             else:
                print(f"  No projects found for entity '{entity}'.")
        else:
            print(f"  No projects data or unexpected format received for entity '{entity}'. Response: {projects_data}")
            
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while fetching projects: {http_err}")
        print(f"Response content: {response.text}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred while fetching projects: {req_err}")
    except json.JSONDecodeError:
        print(f"Failed to decode JSON response from projects endpoint. Response: {response.text}")

if __name__ == '__main__':
    print("Attempting to fetch W&B information...")
    user_entity = get_user_info()
    if user_entity:
        get_projects(user_entity)
    else:
        print("\nCould not determine user entity. Please check API key and permissions.")
    print("\nScript finished.")
