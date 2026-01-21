# 🏥 Healthcare Patient Management System

A comprehensive healthcare management system integrating **Semantic Kernel**, **MCP (Model Context Protocol)**, **SQLite**, **LLM (Ollama)**, and **Streamlit UI**.

## 🎯 Features

### Core Functionality

- ✅ **Patient Management**: Complete patient records with demographics and medical history
- 📋 **Medical Records**: Track diagnoses, treatments, and prescriptions
- 📅 **Appointment Scheduling**: Schedule and manage patient appointments
- 💊 **Prescription Management**: Track medications and check drug interactions
- 🩺 **Vital Signs Monitoring**: Record and visualize patient vitals over time
- 📊 **Analytics Dashboard**: System-wide statistics and insights

### AI-Powered Features

- 🤖 **AI Healthcare Assistant**: Natural language queries using MCP tools
- 🧠 **Symptom Analysis**: AI-powered differential diagnosis suggestions
- 📝 **Medical Summaries**: Automatic generation of patient summaries
- 💡 **Treatment Planning**: AI-assisted treatment plan suggestions
- ⚠️ **Drug Interaction Checking**: Automated safety checks for medications
- 📈 **Vitals Analysis**: Intelligent analysis of vital sign trends

## 🏗️ Architecture

```
healthcare_system/
├── Streamlit UI Layer
│   └── Interactive web interface
├── Semantic Kernel Layer
│   ├── Medical Analysis Plugin
│   ├── Diagnosis Assistant Plugin
│   └── AI Orchestration
├── Agent Layer (LangChain + MCP)
│   ├── Healthcare Agent (ReAct)
│   └── MCP Client
├── MCP Server Layer
│   ├── Medical Server
│   │   ├── Patient management
│   │   ├── Appointments
│   │   └── Medical records
│   └── Pharmacy Server
│       ├── Medication database
│       ├── Drug interactions
│       └── Prescriptions
└── Data Layer
    └── SQLite Database
```

## 📋 Prerequisites

- Python 3.10 or higher
- Ollama installed and running locally
- Ollama model downloaded: `ollama pull llama3.2:3b`

## 🚀 Installation

### 1. Clone or Create Project Structure

Create the folder structure as shown in the project tree.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and configure your settings:

```env
# Ollama Configuration
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434

# Database
DATABASE_PATH=database/healthcare.db

# Optional: OpenAI (for Semantic Kernel)
OPENAI_API_KEY=your_api_key_here

# Application Settings
DEBUG_MODE=True
MAX_TOKENS=1000
TEMPERATURE=0.3
```

### 4. Initialize Database

```bash
python database/schema.py
```

This will create the SQLite database with sample data.

## 🎮 Usage

### Running the Complete System

#### 1. Start Ollama (if not already running)

```bash
ollama serve
```

#### 2. Launch the Streamlit Application

```bash
streamlit run ui/app.py
```

The application will open in your browser at `http://localhost:8501`

### Testing Individual Components

#### Test MCP Medical Server

```bash
python servers/medical_server.py
```

#### Test MCP Pharmacy Server

```bash
python servers/pharmacy_server.py
```

#### Test Healthcare Agent

```bash
python agents/healthcare_agent.py
```

#### Test Semantic Kernel

```bash
python semantic_kernel/kernel_setup.py
```

#### Test Database Operations

```bash
python database/operations.py
```

## 📱 Using the UI

### 1. Dashboard

- View system statistics
- See upcoming appointments
- Quick access to recent patients

### 2. Patient Management

- **Search Patients**: Find patients by name or email
- **Patient Details**: View complete patient information
- **Add New Patient**: Register new patients

### 3. Medical Records

- View complete medical history
- Add new medical records
- Track diagnoses and treatments

### 4. Appointments

- View and filter appointments
- Schedule new appointments
- Update appointment status

### 5. Prescriptions

- View patient prescriptions
- Check drug interactions
- Track medication adherence

### 6. Vital Signs

- Record vital signs (BP, HR, Temp, SpO2)
- View vitals history
- Visualize trends over time

### 7. AI Assistant

Ask natural language questions like:

- "What are the current system statistics?"
- "Search for patients named 'John'"
- "Get details for patient ID 1"
- "What medications is patient 1 currently taking?"
- "Check for drug interactions if we prescribe Aspirin to patient 2"
- "Show me appointments for this week"

### 8. Medical Analysis (Semantic Kernel)

- **Symptom Analysis**: Get AI-powered analysis of symptoms
- **Medical Summaries**: Generate comprehensive patient summaries
- **Treatment Planning**: Get AI-assisted treatment suggestions
- **Vitals Analysis**: Analyze vital signs with AI

## 🔧 Technical Details

### Semantic Kernel Integration

The system uses Semantic Kernel for:

- Medical analysis and insights
- Symptom evaluation
- Treatment plan generation
- Medical summary creation

### MCP (Model Context Protocol)

Two MCP servers provide tool access:

**Medical Server Tools:**

- `search_patient`: Search for patients
- `get_patient_details`: Get complete patient information
- `get_medical_history`: Retrieve medical records
- `get_appointments`: Fetch appointments
- `schedule_appointment`: Create new appointments
- `get_prescriptions`: Get prescription list
- `get_vitals_history`: Retrieve vital signs
- `record_vitals`: Record new vital signs
- `get_system_statistics`: Get system stats

**Pharmacy Server Tools:**

- `search_medications`: Search medication database
- `get_medication_info`: Get detailed drug information
- `check_drug_interactions`: Check for interactions
- `add_prescription`: Create new prescription
- `get_drug_info_online`: Search online drug databases
- `check_medication_adherence`: Analyze adherence

### LangChain ReAct Agent

The healthcare agent uses:

- LangChain's ReAct pattern for reasoning
- Ollama (llama3.2:3b) as the LLM
- MCP tools for database operations
- Multi-turn conversations with full history

### Database Schema

**Tables:**

- `patients`: Patient demographics
- `medical_records`: Diagnoses and treatments
- `appointments`: Scheduled appointments
- `medications`: Medication database
- `prescriptions`: Active and past prescriptions
- `lab_results`: Laboratory test results
- `vitals`: Vital signs measurements

## 🎨 Customization

### Adding New MCP Tools

1. Edit `servers/medical_server.py` or `servers/pharmacy_server.py`
2. Add a new function with `@mcp.tool()` decorator
3. Restart the MCP servers

### Adding New Semantic Kernel Plugins

1. Edit `semantic_kernel/plugins.py`
2. Create a new class with `@sk_function` decorated methods
3. Register the plugin in your kernel

### Modifying the UI

1. Edit `ui/app.py` for page structure
2. Edit `ui/components.py` for reusable components
3. Use Streamlit's hot reload for instant updates

## 🐛 Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve
```

### Database Issues

```bash
# Reinitialize database
python database/schema.py
```

### MCP Server Issues

Check that the server paths in `config/settings.py` are correct.

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 📊 Sample Data

The system includes sample data:

- 4 patients
- 3 medical records
- 4 medications
- Sample appointments and prescriptions

## 🔐 Security Notes

⚠️ **Important**: This is a demonstration system. For production use:

- Implement proper authentication and authorization
- Encrypt sensitive medical data
- Use HTTPS for all communications
- Implement audit logging
- Follow HIPAA compliance requirements
- Use environment-specific configurations

## 🤝 Contributing

This is a demonstration project showing integration of:

- Semantic Kernel
- MCP (Model Context Protocol)
- SQLite
- LLM (Ollama)
- Streamlit

Feel free to extend and customize for your needs.

## 📝 License

This project is provided as-is for educational and demonstration purposes.

## 🙏 Acknowledgments

- **Semantic Kernel** by Microsoft
- **MCP** by Anthropic
- **LangChain** for agent orchestration
- **Ollama** for local LLM inference
- **Streamlit** for the UI framework

## 📧 Support

For issues or questions about:

- **Semantic Kernel**: https://github.com/microsoft/semantic-kernel
- **MCP**: https://modelcontextprotocol.io
- **Ollama**: https://ollama.ai
- **Streamlit**: https://streamlit.io

---

**Built with ❤️ using Semantic Kernel + MCP + SQLite + LLM + Streamlit**
