public class Flow {
    private String flowPath;
    private String eventPath;
    private String npyPath;
    private String npyName;

    public String getNpyName() {
        return npyName;
    }

    public void setNpyName(String npyName) {
        this.npyName = npyName;
    }

    public Flow() {
    }

    @Override
    public String toString() {
        return "Flow{" +
                "flowPath='" + flowPath + '\'' +
                ", eventPath='" + eventPath + '\'' +
                ", npyPath='" + npyPath + '\'' +
                ", npyName='" + npyName + '\'' +
                '}';
    }

    public Flow(String flowPath, String eventPath, String npyPath, String npyName) {
        this.flowPath = flowPath;
        this.eventPath = eventPath;
        this.npyPath = npyPath;
        this.npyName = npyName;
    }

    public Flow(String flowPath, String eventPath, String npyPath) {
        this.flowPath = flowPath;
        this.eventPath = eventPath;
        this.npyPath = npyPath;
    }

    public String getFlowPath() {
        return flowPath;
    }

    public void setFlowPath(String flowPath) {
        this.flowPath = flowPath;
    }

    public String getEventPath() {
        return eventPath;
    }

    public void setEventPath(String eventPath) {
        this.eventPath = eventPath;
    }

    public String getNpyPath() {
        return npyPath;
    }

    public void setNpyPath(String npyPath) {
        this.npyPath = npyPath;
    }

}
