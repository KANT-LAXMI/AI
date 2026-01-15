from openpyxl import Workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
import os

wb = Workbook()
ws = wb.active
ws.title = "All_API_Endpoints"

data = {
    "User_Management": [
        ["S.No", "API Type", "API Link", "Payload", "Response"],
        [1, "GET", "/api/user/get_roles", "None", '[{"RoleID": int, "RoleName": string}]'],
        [2, "POST", "/api/user/create_user", '{"first_name": string, "last_name": string, "email": string, "password": string, "role_id": int, "status": boolean}', '{"message": string, "UserId": int}'],
        [3, "GET", "/api/user/getAllUsers", "None", '[{"UserId": int, "FirstName": string, "LastName": string, "Email": string, "CreatedOn": datetime, "IsDeleted": boolean}]'],
        [4, "GET", "/api/user/getAllUsersExclude/<int:user_id>", "None", '[{"UserId": int, "FirstName": string, "LastName": string, "Email": string, "CreatedOn": datetime, "IsDeleted": boolean}]'],
        [5, "POST", "/api/user/login", '{"email": string, "password": string}', '{"UserId": int, "FirstName": string, "LastName": string, "RoleId": int, "RoleName": string}'],
        [6, "PUT", "/api/user/update_user/<int:user_id>", '{"first_name": string, "last_name": string, "email": string, "password": string (optional), "role_id": int, "status": boolean}', '{"message": string}'],
        [7, "DELETE", "/api/user/delete_user/<int:user_id>", "None", '{"message": string}'],
        [8, "GET", "/api/user/all_users", "None", '{"users": [{"UserId": int, "FirstName": string, "LastName": string, "Email": string, "RoleName": string, "Status": string}],  "count": int}'],
    ],
    "Agent_Management": [
        ["S.No", "API Type", "API Link", "Payload", "Response"],
        [1, "POST", "/api/user/addAgent", '{"name": string, "description": string, "createdBy": string, "systemPrompt": string}', '{"message": string}'],
        [2, "GET", "/api/user/getAgents", "None", '{"agents": [{"id": int, "name": string, "description": string, "createdBy": string, "createdAt": datetime, "modifiedBy": string, "modifiedAt": datetime, "isActive": boolean, "isDeleted": boolean, "systemPromptConfiguration": string}], "count": int}'],
        [3, "DELETE", "/api/user/delete_agent/<int:agent_id>", "None", '{"message": string}'],
        [4, "PUT", "/api/user/updateAgent/<int:agent_id>", '{"name": string, "description": string, "modifiedBy": string, "systemPrompt": string, "isActive": boolean}', '{"message": string}'],
    ],
    "SubAgent_Management": [
        ["S.No", "API Type", "API Link", "Payload", "Response"],
        [1, "POST", "/api/user/addSubAgent", '{"agent_id": int, "name": string, "description": string, "createdBy": string, "isActive": boolean}', '{"message": string}'],
        [2, "GET", "/api/user/getSubAgents", "None", '{"subAgents": [{"id": int, "agent_id": int, "name": string, "description": string, "createdBy": string, "createdAt": datetime, "modifiedBy": string, "modifiedAt": datetime, "isActive": boolean, "isDeleted": boolean}],  "count": int}'],
        [3, "PUT", "/api/user/updateSubAgent/<int:sub_agent_id>", '{"agent_id": int, "name": string, "description": string, "modifiedBy": string, "isActive": boolean, "isDeleted": boolean}', '{"message": string}'],
        [4, "DELETE", "/api/user/deleteSubAgent/<int:sub_agent_id>", "None", '{"message": string}'],
    ],
    "Prompt_Management": [
        ["S.No", "API Type", "API Link", "Payload", "Response"],
        [1, "GET", "/api/user/getPrompts", "None", '{"prompts": [{"id": int, "name": string, "description": string, "temperature": float, "template_text": string, "createdBy": string, "createdAt": datetime, "modifiedBy": string, "modifiedAt": datetime, "isActive": boolean, "isDeleted": boolean, "sub_agents_id": int, "sub_agents_name": string}],  "count": int}'],
        [2, "POST", "/api/user/addPrompt", '{"name": string, "description": string, "temperature": float, "template_text": string, "createdBy": string, "subAgentId": int}', '{"message": string, "prompt_id": int}'],
        [3, "DELETE", "/api/user/deletePrompt/<int:prompt_id>", "None", '{"message": string}'],
        [4, "PUT", "/api/user/updatePrompt/<int:prompt_id>", '{"name": string, "description": string, "temperature": float, "template_text": string, "modifiedBy": string, "isActive": boolean, "isDeleted": boolean, "subAgentId": int}', '{"message": string}'],
    ],
    "Prompt_Variable_Management": [
        ["S.No", "API Type", "API Link", "Payload", "Response"],
        [1, "POST", "/api/user/addPromptVariable", '{"prompt_id": int, "variable_name": string, "variable_type": string, "default_value": string, "required": boolean, "createdBy": string}', '{"message": string}'],
        [2, "GET", "/api/user/getPromptVariables/<int:prompt_id>", "None", '{"prompt_name": string,"variables": [{"id": int, "variable_name": string, "variable_type": string, "default_value": string, "required": boolean, "createdBy": string}],"count": int}'],
        [3, "DELETE", "/api/user/deletePromptVariable/<int:variable_id>", "None", '{"message": string}'],
        [4, "PUT", "/api/user/updatePromptVariable/<int:variable_id>", '{"variable_name": string, "variable_type": string, "default_value": string, "required": boolean, "updatedBy": string}', '{"message": string}'],
    ],
    "Document_Project": [
        ["S.No", "API Type", "API Link", "Payload", "Response"],
        [1, "GET", "/api/user/get_user_history", "user_id", '{"status": string, "data": array}'],
        [2, "GET", "/api/user/get_user_documents", "user_id", '{"status": string, "data": array}'],
        [3, "GET", "/api/user/get_user_documents_by_project", "user_id, project_id", '{"status": string, "data": array}'],
        [4, "GET", "/api/user/get_related_document", "related_id", '{"status": string, "download_link": string}'],
        [5, "GET", "/api/user/get_projects", "user_id", '{"status": string, "data": array}'],
        [6, "POST", "/api/user/userAccess", '{"user_id": int, "client_id": int, "project_id": int, "created_by": string}', '{"message": string}'],
    ],
    "Dashboard_Analytics": [
        ["S.No", "API Type", "API Link", "Payload", "Response"],
        [1, "GET", "/api/user/top_users", "None", '{"users": [{"name": string, "documents": int, "lastActive": string, "avatar": string}], "count": int}'],
        [2, "GET", "/api/user/metrics", "None", '{"metrics": [{"label": string, "value": int/string, "change": string, "subtext": string, "icon": string, "color": string, "alert": boolean}], "peak_time":{"Peak": string}, "resolution_rate": float}'],
        [3, "GET", "/api/user/agent_usage", "None", '{"agent_usage": object}'],
        [4, "GET", "/api/user/top_agents", "None", '{"users": [{"role": string, "documents": int}], "count": int}'],
        [5, "GET", "/api/user/feedback_summary_by_role", "None", '{"feedback_summary":[{"role": string, "positive": int, "neutral": int, "negative": int, "total_feedback": int}], "count": int}'],
    ],
    "Feedback": [
        ["S.No", "API Type", "API Link", "Payload", "Response"],
        [1, "GET", "/api/user/all_feedbacks", "None", '{"feedbacks":[{"userName": string, "feedbackText": string, "rating": int, "feedbackType": string, "createdOn": string}],  "count": int}'],
    ],
}

first = True 
for sheet_name, rows in data.items():
    if first: 
        ws = wb.active 
        ws.title = sheet_name 
        first = False 
    else: 
        ws = wb.create_sheet(title=sheet_name) 
        for row in rows: ws.append(row) 
        filepath = os.path.join(os.getcwd(), "New_SDLC_Admin_Dashboard_Complete.xlsx")
        wb.save(filepath) 