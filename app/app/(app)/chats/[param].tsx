import { StyleSheet, Text, View } from "react-native";
import React from "react";
import { useLocalSearchParams } from "expo-router";

export default function Chat() {
    const { param } = useLocalSearchParams();
    return (
        <View>
            <Text>{param}</Text>
        </View>
    );
}

const styles = StyleSheet.create({});
