#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

int main(int argc, char** argv) {
    ros::init(argc, argv, "my_image_publisher");
    ros::NodeHandle nh;

    if (argc != 2) {
        ROS_ERROR("Usage: rosrun my_image_publisher my_image_publisher <directory>");
        return 1;
    }

    std::string folder_path = argv[1];

    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("/camera/image", 1);

    double publish_rate = 10.0;  // 10Hz
    ros::Rate rate(publish_rate);

    std::vector<std::string> image_files;
    for (auto& entry : fs::directory_iterator(folder_path)) {
        image_files.push_back(entry.path().string());
    }
    std::sort(image_files.begin(), image_files.end());

    size_t image_count = image_files.size();
    size_t current_image = 0;

    if (image_count == 0) {
        ROS_ERROR("No images found in the specified folder.");
        return 1;
    }

    while (ros::ok()) {
        cv::Mat image = cv::imread(image_files[current_image], cv::IMREAD_COLOR);
        if (image.empty()) {
            ROS_ERROR("Failed to read image from %s", image_files[current_image].c_str());
            break;
        }

        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
        pub.publish(msg);

        ROS_INFO("Published image %s", image_files[current_image].c_str());

        current_image = (current_image + 1) % image_count;
        rate.sleep();
    }

    return 0;
}
