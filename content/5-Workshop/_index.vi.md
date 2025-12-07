---
title: "Workshop"
weight: 5
chapter: false
pre: " <b> 5. </b> "
---

# Xác thực: AWS Amplify & Amazon Cognito

## Tổng quan

Workshop thực hành giới thiệu các nguyên tắc cơ bản về xác thực người dùng với AWS Amplify và Amazon Cognito. Người tham gia xây dựng hệ thống xác thực hoàn chỉnh bao gồm đăng ký người dùng, xác minh email, đăng nhập/đăng xuất, quản lý mật khẩu, xác thực đa yếu tố (MFA), và đăng nhập mạng xã hội. Tập trung vào triển khai xác thực an toàn nhanh chóng mà không cần quản lý hạ tầng, tuân theo các best practices của AWS.

## Nội dung chương trình

- Nền tảng xác thực: tại sao chọn Amplify & Cognito, best practices bảo mật, các trường hợp sử dụng.
- Bắt đầu: cài đặt Amplify CLI, khởi tạo ứng dụng React, thêm danh mục xác thực.
- Đăng ký người dùng: xây dựng quy trình đăng ký, xác minh email, thuộc tính tùy chỉnh, chính sách mật khẩu.
- Đăng nhập & phiên làm việc: triển khai xác thực, xử lý JWT tokens, quản lý phiên người dùng, đăng xuất đúng cách.
- Quản lý mật khẩu: quy trình quên mật khẩu, đặt lại mật khẩu, thay đổi mật khẩu, yêu cầu bảo mật.
- Tính năng nâng cao: kích hoạt MFA (TOTP), thêm đăng nhập mạng xã hội (Google/Facebook), nhóm người dùng & vai trò.
- Triển khai: host với Amplify Hosting, cấu hình CI/CD, tên miền tùy chỉnh, giám sát.

## Những gì bạn sẽ xây dựng

- Ứng dụng React với giao diện xác thực hoàn chỉnh (đăng ký, đăng nhập, hồ sơ).
- Amazon Cognito User Pool với xác minh email và chính sách mật khẩu tùy chỉnh.
- Xác thực đa yếu tố sử dụng TOTP (Time-based One-Time Password).
- Tích hợp đăng nhập mạng xã hội với Google và Facebook.
- Routes được bảo vệ với kiểm soát truy cập dựa trên vai trò (RBAC).

## Yêu cầu

- Tài khoản AWS có quyền truy cập console.
- Node.js 18+ và npm đã cài đặt.
- Hiểu biết cơ bản về JavaScript/React.
- Amplify CLI đã cài đặt (`npm install -g @aws-amplify/cli`).

## Tài liệu tham khảo

- Các bước cấu hình Amplify CLI
- Hướng dẫn thiết lập Cognito User Pool
- Tích hợp nhà cung cấp danh tính mạng xã hội
- Danh sách kiểm tra dọn dẹp